// Integration tests for the C API on the Metal GPU, driven purely through
// MLX-C the way a non-C++ caller (e.g. Rust via mlx-bridge) would.

#include "json_value.h"
#include "mlx_c_interop.h"
#include "mlx_perf/mlx_perf_c_api.h"
#include "test_harness.h"

#include "mlx/c/array.h"
#include "mlx/c/device.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"

#include <cmath>
#include <fstream>
#include <map>
#include <string>

namespace {

struct TwoStreamPayload {
  mlx_stream side_stream;
  mlx_stream main_stream;
  int build_count = 0;
};

// matmul on a side stream, exp on the main stream: one cross-stream join.
int build_two_stream_graph(mlx_vector_array fresh_outputs, void* builder_payload) {
  auto* payload = static_cast<TwoStreamPayload*>(builder_payload);
  ++payload->build_count;
  const int shape[] = {128, 128};
  mlx_array square_ones = mlx_array_new();
  mlx_array product = mlx_array_new();
  mlx_array activated = mlx_array_new();
  int status = mlx_ones(&square_ones, shape, 2, MLX_FLOAT32, payload->side_stream);
  status |= mlx_matmul(&product, square_ones, square_ones, payload->side_stream);
  status |= mlx_exp(&activated, product, payload->main_stream);
  status |= mlx_vector_array_append_value(fresh_outputs, activated);
  mlx_array_free(square_ones);
  mlx_array_free(product);
  mlx_array_free(activated);
  return status;
}

// About ten primitives on two streams: a side branch (ones, matmul, exp,
// sin) and a main branch (ones, cos) joined across streams, then three more
// steps on the main stream. Enough nodes in a row on each stream to have
// gaps between consecutive GPU executions.
int build_ten_node_graph(mlx_vector_array fresh_outputs, void* builder_payload) {
  auto* payload = static_cast<TwoStreamPayload*>(builder_payload);
  ++payload->build_count;
  const int shape[] = {64, 64};
  mlx_array node_handles[10];
  for (mlx_array& node_handle : node_handles) {
    node_handle = mlx_array_new();
  }
  int status = mlx_ones(&node_handles[0], shape, 2, MLX_FLOAT32, payload->side_stream);
  status |= mlx_matmul(&node_handles[1], node_handles[0], node_handles[0], payload->side_stream);
  status |= mlx_exp(&node_handles[2], node_handles[1], payload->side_stream);
  status |= mlx_sin(&node_handles[3], node_handles[2], payload->side_stream);
  status |= mlx_ones(&node_handles[4], shape, 2, MLX_FLOAT32, payload->main_stream);
  status |= mlx_cos(&node_handles[5], node_handles[4], payload->main_stream);
  status |= mlx_add(&node_handles[6], node_handles[3], node_handles[5], payload->main_stream);
  status |= mlx_multiply(&node_handles[7], node_handles[6], node_handles[6], payload->main_stream);
  status |= mlx_tanh(&node_handles[8], node_handles[7], payload->main_stream);
  status |= mlx_exp(&node_handles[9], node_handles[8], payload->main_stream);
  status |= mlx_vector_array_append_value(fresh_outputs, node_handles[9]);
  for (mlx_array& node_handle : node_handles) {
    mlx_array_free(node_handle);
  }
  return status;
}

// One primitive (ones) on the main stream: the smallest measurable job.
int build_single_node_graph(mlx_vector_array fresh_outputs, void* builder_payload) {
  auto* payload = static_cast<TwoStreamPayload*>(builder_payload);
  ++payload->build_count;
  const int shape[] = {16};
  mlx_array single_output = mlx_array_new();
  int status = mlx_ones(&single_output, shape, 1, MLX_FLOAT32, payload->main_stream);
  status |= mlx_vector_array_append_value(fresh_outputs, single_output);
  mlx_array_free(single_output);
  return status;
}

// The two-stream matmul, finished on the main stream by a caller-chosen
// unary MLX-C operation: two implementations with equal output shapes.
struct UnaryVariantPayload {
  TwoStreamPayload streams;
  int (*finishing_operation)(mlx_array*, const mlx_array, const mlx_stream);
};

int build_unary_variant_graph(mlx_vector_array fresh_outputs, void* builder_payload) {
  auto* payload = static_cast<UnaryVariantPayload*>(builder_payload);
  ++payload->streams.build_count;
  const int shape[] = {128, 128};
  mlx_array square_ones = mlx_array_new();
  mlx_array product = mlx_array_new();
  mlx_array finished = mlx_array_new();
  int status = mlx_ones(&square_ones, shape, 2, MLX_FLOAT32, payload->streams.side_stream);
  status |= mlx_matmul(&product, square_ones, square_ones, payload->streams.side_stream);
  status |= payload->finishing_operation(&finished, product, payload->streams.main_stream);
  status |= mlx_vector_array_append_value(fresh_outputs, finished);
  mlx_array_free(square_ones);
  mlx_array_free(product);
  mlx_array_free(finished);
  return status;
}

int build_nothing(mlx_vector_array, void*) { return 0; }
int build_failure(mlx_vector_array, void*) { return 7; }

struct ApiFixture {
  TwoStreamPayload payload{mlx_stream_new_device(mlx_device_new_type(MLX_GPU, 0)),
                           mlx_default_gpu_stream_new()};
  ~ApiFixture() {
    mlx_stream_free(payload.side_stream);
    mlx_stream_free(payload.main_stream);
  }
};

// Two variants on their own streams, sin and cos: same shapes, other values.
struct CompareFixture {
  ApiFixture sin_streams;
  ApiFixture cos_streams;
  UnaryVariantPayload sin_payload{sin_streams.payload, mlx_sin};
  UnaryVariantPayload cos_payload{cos_streams.payload, mlx_cos};
  mlx_perf_variant sin_variant{"sin", build_unary_variant_graph, &sin_payload};
  mlx_perf_variant cos_variant{"cos", build_unary_variant_graph, &cos_payload};
};

// --- positive ---

void should_measure_an_mlx_c_job_and_write_its_report() {
  ApiFixture fixture;
  mlx_perf_job_summary summary{};
  const int status = mlx_perf_profile_to_report(
      build_two_stream_graph, &fixture.payload, "c-api", "C API job", ".",
      "c_api_test_report", &summary);
  EXPECT_TRUE(status == 0);
  EXPECT_TRUE(std::string(mlx_perf_last_error()).empty());
  EXPECT_TRUE(std::ifstream("c_api_test_report.html").good());
  EXPECT_TRUE(summary.executed_node_count > 0);
  EXPECT_TRUE(summary.gpu_measured_node_count == summary.executed_node_count);
  EXPECT_TRUE(summary.stream_count == 2);
  EXPECT_TRUE(summary.job_wall_min_us <= summary.job_wall_median_us &&
              summary.job_wall_median_us <= summary.job_wall_max_us);
  // stream probe + warm-up + 7 samples + timeline + serial = 11 fresh graphs.
  EXPECT_TRUE(fixture.payload.build_count == 11);
}

void should_read_mlx_c_handles_as_the_same_core_arrays() {
  // Pins the one MLX-C layout assumption mlx_c_interop relies on.
  mlx_vector_array handles = mlx_vector_array_new();
  mlx_array three = mlx_array_new_float(3.0f);
  mlx_vector_array_append_value(handles, three);
  const auto core_arrays = mlx_perf::read_mlx_c_vector(handles);
  EXPECT_TRUE(core_arrays.size() == 1);
  EXPECT_TRUE(core_arrays[0].item<float>() == 3.0f);
  mlx_array_free(three);
  mlx_vector_array_free(handles);
  // Still valid after every MLX-C handle is gone: the copy shares data.
  EXPECT_TRUE(core_arrays[0].item<float>() == 3.0f);
}

void should_report_the_library_version_the_header_names() {
  EXPECT_TRUE(std::string(mlx_perf_version()) == MLX_PERF_VERSION_STRING);
  EXPECT_TRUE(std::string(mlx_perf_version()) ==
              std::to_string(MLX_PERF_VERSION_MAJOR) + "." +
                  std::to_string(MLX_PERF_VERSION_MINOR) + "." +
                  std::to_string(MLX_PERF_VERSION_PATCH));
}

// #11's acceptance: from the file alone, each node's GPU execution time and
// the gap between one node's GPU end and the next node's GPU start.
void should_write_per_node_measurements_a_caller_can_read_without_the_report() {
  ApiFixture fixture;
  mlx_perf_job_summary summary{};
  EXPECT_TRUE(mlx_perf_profile_to_measurements(build_ten_node_graph, &fixture.payload,
                                               "ten-node", ".", "c_api_measurements",
                                               &summary) == 0);
  EXPECT_TRUE(!std::ifstream("c_api_measurements.html").good());
  const mlx_perf::JsonValue document = mlx_perf::parse_json_file("c_api_measurements.json");
  const auto& node_records = document.at("nodes").as_array();
  EXPECT_TRUE(static_cast<int>(node_records.size()) == summary.executed_node_count);
  EXPECT_TRUE(node_records.size() >= 10);
  std::map<double, const mlx_perf::JsonValue*> stream_to_previous_node;
  int checked_gap_count = 0;
  for (const mlx_perf::JsonValue& node_record : node_records) {
    const double gpu_started = node_record.at("gpu_started_us").as_number();
    const double gpu_finished = node_record.at("gpu_finished_us").as_number();
    EXPECT_TRUE(gpu_finished >= gpu_started);
    EXPECT_TRUE(node_record.at("is_compute_gpu_measured").as_bool());
    const double stream_index = node_record.at("stream_index").as_number();
    const auto previous_entry = stream_to_previous_node.find(stream_index);
    if (previous_entry != stream_to_previous_node.end()) {
      const double expected_gap =
          gpu_started - previous_entry->second->at("gpu_finished_us").as_number();
      EXPECT_TRUE(std::abs(node_record.at("gpu_idle_before_us").as_number() - expected_gap) <
                  1e-6);
      ++checked_gap_count;
    } else {
      EXPECT_TRUE(node_record.at("gpu_idle_before_us").is_null());
    }
    stream_to_previous_node[stream_index] = &node_record;
  }
  EXPECT_TRUE(checked_gap_count >= 8);
}

void should_account_for_every_call_into_the_build_callback() {
  ApiFixture fixture;
  EXPECT_TRUE(mlx_perf_profile_to_measurements(build_ten_node_graph, &fixture.payload,
                                               "calls", ".", "c_api_build_calls",
                                               nullptr) == 0);
  const mlx_perf::JsonValue document = mlx_perf::parse_json_file("c_api_build_calls.json");
  const auto& build_records = document.at("job").at("graph_build").as_array();
  EXPECT_TRUE(build_records.size() == 5);
  EXPECT_TRUE(build_records[0].at("pass").as_string() == "stream_discovery");
  std::size_t recorded_call_count = 0;
  for (const mlx_perf::JsonValue& build_record : build_records) {
    recorded_call_count += build_record.at("calls_us").as_array().size();
  }
  EXPECT_TRUE(static_cast<int>(recorded_call_count) == fixture.payload.build_count);
  EXPECT_TRUE(document.at("job").at("job_wall_us").at("samples").as_array().size() == 7);
}

void should_write_the_measurements_beside_the_report() {
  ApiFixture fixture;
  EXPECT_TRUE(mlx_perf_profile_to_report(build_two_stream_graph, &fixture.payload, "c-api",
                                         "C API job", ".", "c_api_report_and_data",
                                         nullptr) == 0);
  EXPECT_TRUE(std::ifstream("c_api_report_and_data.html").good());
  const mlx_perf::JsonValue document =
      mlx_perf::parse_json_file("c_api_report_and_data.json");
  EXPECT_TRUE(document.at("variant_name").as_string() == "c-api");
}

void should_compare_two_mlx_c_jobs_and_write_the_verdict_page() {
  CompareFixture fixture;
  mlx_perf_comparison_summary summary{};
  EXPECT_TRUE(mlx_perf_compare_to_report(&fixture.sin_variant, &fixture.cos_variant,
                                         "sin vs cos", ".", "c_api_compare", &summary) == 0);
  for (const char* written_file :
       {"c_api_compare.html", "c_api_compare.json", "c_api_compare_sin.html",
        "c_api_compare_cos.html", "c_api_compare_sin.json", "c_api_compare_cos.json"}) {
    EXPECT_TRUE(std::ifstream(written_file).good());
  }
  for (const mlx_perf_metric_comparison* metric_comparison :
       {&summary.job_wall, &summary.gpu_span, &summary.host_submission}) {
    EXPECT_TRUE(metric_comparison->round_count == 12);
    EXPECT_TRUE(metric_comparison->difference_interval_coverage > 0.95);
    // The verdict is exactly what its own interval says.
    const int verdict_from_interval =
        metric_comparison->difference_interval_high_us < 0.0  ? MLX_PERF_VERDICT_CANDIDATE_FASTER
        : metric_comparison->difference_interval_low_us > 0.0 ? MLX_PERF_VERDICT_CANDIDATE_SLOWER
                                                              : MLX_PERF_VERDICT_WITHIN_SPREAD;
    EXPECT_TRUE(metric_comparison->verdict == verdict_from_interval);
    EXPECT_TRUE(metric_comparison->sign_test_p_value > 0.0 &&
                metric_comparison->sign_test_p_value <= 1.0);
  }
  EXPECT_TRUE(summary.job_wall.baseline_min_us <= summary.job_wall.baseline_median_us);
  EXPECT_TRUE(summary.max_absolute_output_difference > 0.0);  // sin and cos differ
  EXPECT_TRUE(summary.baseline.stream_count == 2);
  // Stream discovery 1, warm-up 1, 12 uninstrumented and 12 GPU-span rounds,
  // one timeline and one serial pass.
  EXPECT_TRUE(fixture.sin_payload.streams.build_count == 28);
  EXPECT_TRUE(fixture.cos_payload.streams.build_count == 28);
  const mlx_perf::JsonValue record = mlx_perf::parse_json_file("c_api_compare.json");
  EXPECT_TRUE(record.at("candidate").at("variant_name").as_string() == "cos");
}

// --- edge / boundary ---

void should_leave_the_gap_empty_for_a_single_node_job() {
  ApiFixture fixture;
  EXPECT_TRUE(mlx_perf_profile_to_measurements(build_single_node_graph, &fixture.payload,
                                               "single", ".", "c_api_single_node",
                                               nullptr) == 0);
  const mlx_perf::JsonValue document = mlx_perf::parse_json_file("c_api_single_node.json");
  const auto& node_records = document.at("nodes").as_array();
  EXPECT_TRUE(!node_records.empty());
  EXPECT_TRUE(node_records.front().at("gpu_idle_before_us").is_null());
}

void should_leave_the_last_error_untouched_when_asked_for_the_version() {
  EXPECT_TRUE(mlx_perf_profile_to_report(build_failure, nullptr, "v", "t", ".", "x",
                                         nullptr) != 0);
  EXPECT_TRUE(mlx_perf_version() != nullptr);
  EXPECT_TRUE(std::string(mlx_perf_last_error()).find("returned 7") != std::string::npos);
}


void should_accept_a_null_summary() {
  ApiFixture fixture;
  EXPECT_TRUE(mlx_perf_profile_to_report(build_two_stream_graph, &fixture.payload,
                                         "c-api", "t", ".", "c_api_null_summary",
                                         nullptr) == 0);
}

void should_reject_a_callback_that_produces_no_outputs() {
  EXPECT_TRUE(mlx_perf_profile_to_report(build_nothing, nullptr, "v", "t", ".",
                                         "x", nullptr) != 0);
  EXPECT_TRUE(std::string(mlx_perf_last_error()).find("no outputs") != std::string::npos);
}

// --- negative ---

void should_surface_a_failing_callback_as_an_error_not_a_crash() {
  EXPECT_TRUE(mlx_perf_profile_to_report(build_failure, nullptr, "v", "t", ".",
                                         "x", nullptr) != 0);
  EXPECT_TRUE(std::string(mlx_perf_last_error()).find("returned 7") != std::string::npos);
}

void should_reject_null_arguments() {
  EXPECT_TRUE(mlx_perf_profile_to_report(nullptr, nullptr, "v", "t", ".", "x",
                                         nullptr) != 0);
  EXPECT_TRUE(mlx_perf_profile_to_report(build_nothing, nullptr, nullptr, "t", ".",
                                         "x", nullptr) != 0);
}

void should_report_an_unwritable_output_directory() {
  ApiFixture fixture;
  EXPECT_TRUE(mlx_perf_profile_to_report(build_two_stream_graph, &fixture.payload, "v",
                                         "t", "no/such/directory", "x",
                                         nullptr) != 0);
  EXPECT_TRUE(std::string(mlx_perf_last_error()).find("cannot write") !=
              std::string::npos);
}

void should_reject_null_arguments_to_the_measurements_entry_point() {
  EXPECT_TRUE(mlx_perf_profile_to_measurements(nullptr, nullptr, "v", ".", "x", nullptr) != 0);
  EXPECT_TRUE(std::string(mlx_perf_last_error()) ==
              "mlx_perf_profile_to_measurements: null argument");
  EXPECT_TRUE(mlx_perf_profile_to_measurements(build_nothing, nullptr, "v", ".", nullptr,
                                               nullptr) != 0);
}

void should_report_an_unwritable_measurements_directory() {
  ApiFixture fixture;
  EXPECT_TRUE(mlx_perf_profile_to_measurements(build_two_stream_graph, &fixture.payload, "v",
                                               "no/such/directory", "x", nullptr) != 0);
  EXPECT_TRUE(std::string(mlx_perf_last_error()).find("cannot write the measurements") !=
              std::string::npos);
}

void should_refuse_to_compare_jobs_with_different_outputs_writing_nothing() {
  CompareFixture fixture;
  mlx_perf_variant single_node_variant{"single", build_single_node_graph,
                                       &fixture.cos_streams.payload};
  EXPECT_TRUE(mlx_perf_compare_to_report(&fixture.sin_variant, &single_node_variant, "t", ".",
                                         "c_api_compare_mismatch", nullptr) != 0);
  EXPECT_TRUE(std::string(mlx_perf_last_error()).find("(128,128)") != std::string::npos);
  EXPECT_TRUE(!std::ifstream("c_api_compare_mismatch.html").good());
}

void should_refuse_variants_whose_names_would_collide_or_escape() {
  CompareFixture fixture;
  mlx_perf_variant same_name_variant{"SIN", build_unary_variant_graph, &fixture.cos_payload};
  EXPECT_TRUE(mlx_perf_compare_to_report(&fixture.sin_variant, &same_name_variant, "t", ".",
                                         "x", nullptr) != 0);
  EXPECT_TRUE(std::string(mlx_perf_last_error()).find("both variants are named") !=
              std::string::npos);
  mlx_perf_variant path_variant{"../cos", build_unary_variant_graph, &fixture.cos_payload};
  EXPECT_TRUE(mlx_perf_compare_to_report(&fixture.sin_variant, &path_variant, "t", ".", "x",
                                         nullptr) != 0);
  EXPECT_TRUE(mlx_perf_compare_to_report(nullptr, &fixture.cos_variant, "t", ".", "x",
                                         nullptr) != 0);
  EXPECT_TRUE(std::string(mlx_perf_last_error()).find("null variant") != std::string::npos);
  mlx_perf_variant empty_name_variant{"", build_unary_variant_graph, &fixture.cos_payload};
  EXPECT_TRUE(mlx_perf_compare_to_report(&fixture.sin_variant, &empty_name_variant, "t", ".",
                                         "x", nullptr) != 0);
  mlx_perf_variant null_name_variant{nullptr, build_unary_variant_graph, &fixture.cos_payload};
  EXPECT_TRUE(mlx_perf_compare_to_report(&fixture.sin_variant, &null_name_variant, "t", ".",
                                         "x", nullptr) != 0);
}

void should_refuse_a_comparison_with_a_null_callback_or_title() {
  CompareFixture fixture;
  mlx_perf_variant no_callback_variant{"cos", nullptr, &fixture.cos_payload};
  EXPECT_TRUE(mlx_perf_compare_to_report(&fixture.sin_variant, &no_callback_variant, "t", ".",
                                         "x", nullptr) != 0);
  EXPECT_TRUE(mlx_perf_compare_to_report(&fixture.sin_variant, &fixture.cos_variant, nullptr,
                                         ".", "x", nullptr) != 0);
  EXPECT_TRUE(std::string(mlx_perf_last_error()).find("null argument") != std::string::npos);
  // Nothing was measured: neither callback was ever called.
  EXPECT_TRUE(fixture.sin_payload.streams.build_count == 0);
}

void should_report_an_unwritable_comparison_directory() {
  CompareFixture fixture;
  EXPECT_TRUE(mlx_perf_compare_to_report(&fixture.sin_variant, &fixture.cos_variant, "t",
                                         "no/such/directory", "x", nullptr) != 0);
  EXPECT_TRUE(std::string(mlx_perf_last_error()).find("cannot write the comparison") !=
              std::string::npos);
}

// The same sizes the Rust example's FFI test asserts with size_of: a field
// added on one side only changes one of the two and fails.
void should_keep_the_c_struct_sizes_the_rust_mirror_expects() {
  EXPECT_TRUE(sizeof(mlx_perf_job_summary) == 72);
  EXPECT_TRUE(sizeof(mlx_perf_metric_comparison) == 144);
  EXPECT_TRUE(sizeof(mlx_perf_comparison_summary) == 584);
  EXPECT_TRUE(sizeof(mlx_perf_variant) == 24);
}

void should_reject_a_null_mlx_c_vector() {
  bool has_thrown = false;
  try {
    mlx_perf::read_mlx_c_vector(mlx_vector_array{nullptr});
  } catch (const std::runtime_error&) {
    has_thrown = true;
  }
  EXPECT_TRUE(has_thrown);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_measure_an_mlx_c_job_and_write_its_report),
      TEST_CASE(should_read_mlx_c_handles_as_the_same_core_arrays),
      TEST_CASE(should_report_the_library_version_the_header_names),
      TEST_CASE(should_write_per_node_measurements_a_caller_can_read_without_the_report),
      TEST_CASE(should_account_for_every_call_into_the_build_callback),
      TEST_CASE(should_write_the_measurements_beside_the_report),
      TEST_CASE(should_leave_the_gap_empty_for_a_single_node_job),
      TEST_CASE(should_compare_two_mlx_c_jobs_and_write_the_verdict_page),
      TEST_CASE(should_leave_the_last_error_untouched_when_asked_for_the_version),
      TEST_CASE(should_accept_a_null_summary),
      TEST_CASE(should_reject_a_callback_that_produces_no_outputs),
      TEST_CASE(should_surface_a_failing_callback_as_an_error_not_a_crash),
      TEST_CASE(should_reject_null_arguments),
      TEST_CASE(should_report_an_unwritable_output_directory),
      TEST_CASE(should_reject_null_arguments_to_the_measurements_entry_point),
      TEST_CASE(should_report_an_unwritable_measurements_directory),
      TEST_CASE(should_refuse_to_compare_jobs_with_different_outputs_writing_nothing),
      TEST_CASE(should_refuse_variants_whose_names_would_collide_or_escape),
      TEST_CASE(should_refuse_a_comparison_with_a_null_callback_or_title),
      TEST_CASE(should_report_an_unwritable_comparison_directory),
      TEST_CASE(should_keep_the_c_struct_sizes_the_rust_mirror_expects),
      TEST_CASE(should_reject_a_null_mlx_c_vector),
  });
}
