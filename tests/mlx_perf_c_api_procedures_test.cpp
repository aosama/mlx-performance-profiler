// Integration tests for the procedure comparison C entry point on the Metal
// GPU, with procedures written purely in MLX-C, as a non-C++ caller (e.g.
// Rust via mlx-bridge) would write them.

#include "json_value.h"
#include "mlx_perf/mlx_perf_c_api.h"
#include "backend_expectations.h"
#include "test_harness.h"

#include "mlx/c/array.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"

#include <fstream>
#include <string>
#include <thread>

namespace {

// `step_count` elementwise steps over a fresh input, evaluated after every
// step or once at the end. A non-zero failing_*_status makes that callback
// return it.
struct SteppedPayload {
  int step_count = 8;
  bool is_evaluating_every_step = false;
  int prepare_count = 0;
  int run_count = 0;
  int failing_prepare_status = 0;
  int failing_run_status = 0;
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  mlx_array step_input = mlx_array_new();
  ~SteppedPayload() {
    mlx_array_free(step_input);
    mlx_stream_free(gpu_stream);
  }
};

int prepare_stepped(void* procedure_payload) {
  auto* payload = static_cast<SteppedPayload*>(procedure_payload);
  ++payload->prepare_count;
  if (payload->failing_prepare_status != 0) return payload->failing_prepare_status;
  const int shape[] = {256, 256};
  int status = mlx_ones(&payload->step_input, shape, 2, MLX_FLOAT32, payload->gpu_stream);
  return status | mlx_array_eval(payload->step_input);
}

int run_stepped(void* procedure_payload) {
  auto* payload = static_cast<SteppedPayload*>(procedure_payload);
  ++payload->run_count;
  if (payload->failing_run_status != 0) return payload->failing_run_status;
  mlx_array running_value = mlx_array_new();
  mlx_array one = mlx_array_new_float(1.0f);
  int status = mlx_array_set(&running_value, payload->step_input);
  for (int step_index = 0; step_index < payload->step_count; ++step_index) {
    status |= mlx_add(&running_value, running_value, one, payload->gpu_stream);
    if (payload->is_evaluating_every_step) status |= mlx_array_eval(running_value);
  }
  status |= mlx_array_eval(running_value);
  mlx_array_free(one);
  mlx_array_free(running_value);
  return status;
}

// #44: the run callback hands the steps to an owner thread and waits. MLX
// streams belong to the thread that made them, so the owner thread runs the
// steps on its own default GPU stream, as an engine's owner thread would.
int run_stepped_on_owner_thread(void* procedure_payload) {
  int owner_thread_status = 0;
  std::thread owner_thread([&owner_thread_status, procedure_payload]() {
    auto* payload = static_cast<SteppedPayload*>(procedure_payload);
    const mlx_stream calling_thread_stream = payload->gpu_stream;
    payload->gpu_stream = mlx_default_gpu_stream_new();
    owner_thread_status = run_stepped(procedure_payload);
    mlx_stream_free(payload->gpu_stream);
    payload->gpu_stream = calling_thread_stream;
  });
  owner_thread.join();
  return owner_thread_status;
}

struct ProcedureFixture {
  SteppedPayload chatty_payload;
  SteppedPayload batched_payload;
  mlx_perf_procedure chatty{"chatty", prepare_stepped, run_stepped, &chatty_payload};
  mlx_perf_procedure batched{"batched", prepare_stepped, run_stepped, &batched_payload};
  ProcedureFixture() { chatty_payload.is_evaluating_every_step = true; }
};

int compare(const ProcedureFixture& fixture, const char* file_base,
            mlx_perf_procedure_comparison_summary* summary) {
  return mlx_perf_compare_procedures_to_report(&fixture.chatty, &fixture.batched, "chatty vs batched", ".",
                                               file_base, summary);
}

bool last_error_mentions(const std::string& expected_text) {
  return std::string(mlx_perf_last_error()).find(expected_text) != std::string::npos;
}

// --- positive ---

void should_compare_two_mlx_c_procedures_and_write_the_page_and_record() {
  ProcedureFixture fixture;
  mlx_perf_procedure_comparison_summary summary{};
  EXPECT_TRUE(compare(fixture, "c_api_procedures", &summary) == 0);
  EXPECT_TRUE(std::ifstream("c_api_procedures.html").good());
  const mlx_perf::JsonValue record = mlx_perf::parse_json_file("c_api_procedures.json");
  EXPECT_TRUE(record.at("kind").as_string() == "procedure_comparison");
  EXPECT_TRUE(summary.baseline.median_blocking_wait_count >= 8);
  EXPECT_TRUE(summary.candidate.median_blocking_wait_count >= 1 && summary.candidate.median_blocking_wait_count <= 2);
  EXPECT_TRUE(summary.baseline.median_command_buffer_count >= 8);
  EXPECT_TRUE(summary.baseline.kernel_launch_count >= 8);
  EXPECT_TRUE(summary.baseline.launches_unable_to_fill_gpu >= -1);
  for (const mlx_perf_metric_comparison* metric_comparison :
       {&summary.duration, &summary.gpu_busy, &summary.gpu_idle}) {
    EXPECT_TRUE(metric_comparison->round_count == 12);
  }
  // Memory on the same rounds (#46): each run evaluates 256 x 256 floats.
  EXPECT_TRUE(summary.peak_mlx_active_added.round_count == 12);
  EXPECT_TRUE(summary.peak_mlx_active_added.baseline_median_bytes > 0);
  EXPECT_TRUE(backend_expectations::HAS_SYSTEM_WIRED_MEMORY ? summary.system_wired_at_end.candidate_median_bytes > 0
                                                         : summary.system_wired_at_end.candidate_median_bytes == 0);
  EXPECT_TRUE(summary.baseline_rounds_crossing_wire_limit >= 0 && summary.baseline_rounds_crossing_wire_limit <= 12);
  // One warm-up, one observed run, 12 rounds: prepare before every run.
  EXPECT_TRUE(fixture.chatty_payload.run_count == 14);
  EXPECT_TRUE(fixture.chatty_payload.prepare_count == 14);
}

void should_report_gpu_work_a_procedure_runs_on_its_owner_thread() {
  ProcedureFixture fixture;
  fixture.batched.run = run_stepped_on_owner_thread;
  mlx_perf_procedure_comparison_summary summary{};
  EXPECT_TRUE(compare(fixture, "c_api_procedures_owner_thread", &summary) == 0);
  EXPECT_TRUE(summary.candidate.median_command_buffer_count >= 1);
  EXPECT_TRUE(summary.candidate.median_blocking_wait_count >= 1);
  EXPECT_TRUE(summary.candidate.kernel_launch_count >= 8);
  EXPECT_TRUE(summary.gpu_busy.candidate_median_us > 0.0);
}

// --- edge / boundary ---

void should_run_a_procedure_without_a_prepare_callback() {
  ProcedureFixture fixture;
  SteppedPayload reused_input_payload;
  prepare_stepped(&reused_input_payload);
  mlx_perf_procedure without_prepare{"no_prepare", nullptr, run_stepped, &reused_input_payload};
  EXPECT_TRUE(mlx_perf_compare_procedures_to_report(&fixture.chatty, &without_prepare, "t", ".",
                                                    "c_api_procedures_no_prepare", nullptr) == 0);
  EXPECT_TRUE(reused_input_payload.prepare_count == 1);
  EXPECT_TRUE(reused_input_payload.run_count == 14);
}

void should_keep_the_procedure_struct_sizes_stable() {
  EXPECT_TRUE(sizeof(mlx_perf_procedure) == 32);
  EXPECT_TRUE(sizeof(mlx_perf_procedure_summary) == 32);
  EXPECT_TRUE(sizeof(mlx_perf_memory_comparison) == 64);
  EXPECT_TRUE(sizeof(mlx_perf_procedure_comparison_summary) == 704);
}

// --- negative ---

void should_surface_a_failing_run_by_procedure_name_writing_nothing() {
  ProcedureFixture fixture;
  fixture.batched_payload.failing_run_status = 7;
  EXPECT_TRUE(compare(fixture, "c_api_procedures_failed_run", nullptr) != 0);
  EXPECT_TRUE(last_error_mentions("procedure batched: run returned 7"));
  EXPECT_TRUE(!std::ifstream("c_api_procedures_failed_run.html").good());
}

void should_surface_a_failing_prepare() {
  ProcedureFixture fixture;
  fixture.chatty_payload.failing_prepare_status = 3;
  EXPECT_TRUE(compare(fixture, "c_api_procedures_failed_prepare", nullptr) != 0);
  EXPECT_TRUE(last_error_mentions("procedure chatty: prepare returned 3"));
  EXPECT_TRUE(fixture.chatty_payload.run_count == 0);
}

void should_refuse_null_procedures_callbacks_and_strings() {
  ProcedureFixture fixture;
  EXPECT_TRUE(mlx_perf_compare_procedures_to_report(nullptr, &fixture.batched, "t", ".", "x", nullptr) != 0);
  EXPECT_TRUE(last_error_mentions("null procedure"));
  mlx_perf_procedure without_run{"no_run", nullptr, nullptr, &fixture.batched_payload};
  EXPECT_TRUE(mlx_perf_compare_procedures_to_report(&fixture.chatty, &without_run, "t", ".", "x", nullptr) != 0);
  EXPECT_TRUE(last_error_mentions("null run callback"));
  mlx_perf_procedure without_name{nullptr, prepare_stepped, run_stepped, &fixture.batched_payload};
  EXPECT_TRUE(mlx_perf_compare_procedures_to_report(&fixture.chatty, &without_name, "t", ".", "x", nullptr) != 0);
  EXPECT_TRUE(mlx_perf_compare_procedures_to_report(&fixture.chatty, &fixture.batched, nullptr, ".", "x",
                                                    nullptr) != 0);
  EXPECT_TRUE(last_error_mentions("mlx_perf_compare_procedures_to_report: null argument"));
  EXPECT_TRUE(fixture.chatty_payload.run_count == 0);
}

void should_refuse_procedure_names_that_would_collide() {
  ProcedureFixture fixture;
  fixture.batched.procedure_name = "CHATTY";
  EXPECT_TRUE(compare(fixture, "x", nullptr) != 0);
  EXPECT_TRUE(last_error_mentions("both procedures are named"));
}

void should_report_an_unwritable_directory() {
  ProcedureFixture fixture;
  EXPECT_TRUE(mlx_perf_compare_procedures_to_report(&fixture.chatty, &fixture.batched, "t", "no/such/directory",
                                                    "x", nullptr) != 0);
  EXPECT_TRUE(last_error_mentions("cannot write the procedure comparison"));
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_compare_two_mlx_c_procedures_and_write_the_page_and_record),
      TEST_CASE(should_report_gpu_work_a_procedure_runs_on_its_owner_thread),
      TEST_CASE(should_run_a_procedure_without_a_prepare_callback),
      TEST_CASE(should_keep_the_procedure_struct_sizes_stable),
      TEST_CASE(should_surface_a_failing_run_by_procedure_name_writing_nothing),
      TEST_CASE(should_surface_a_failing_prepare),
      TEST_CASE(should_refuse_null_procedures_callbacks_and_strings),
      TEST_CASE(should_refuse_procedure_names_that_would_collide),
      TEST_CASE(should_report_an_unwritable_directory),
  });
}
