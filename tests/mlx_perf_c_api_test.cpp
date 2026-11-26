// Integration tests for the C API on the Metal GPU, driven purely through
// MLX-C the way a non-C++ caller (e.g. Rust via mlx-c-rust) would.

#include "mlx_c_interop.h"
#include "mlx_perf_c_api.h"
#include "test_harness.h"

#include "mlx/c/array.h"
#include "mlx/c/device.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"

#include <fstream>
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

// --- positive ---

void should_measure_an_mlx_c_job_and_write_its_report() {
  ApiFixture fixture;
  mlx_perf_job_summary summary{};
  const int status = mlx_perf_profile_to_report(
      build_two_stream_graph, &fixture.payload, "c-api", "C API job", "build",
      "c_api_test_report", &summary);
  EXPECT_TRUE(status == 0);
  EXPECT_TRUE(std::string(mlx_perf_last_error()).empty());
  EXPECT_TRUE(std::ifstream("build/c_api_test_report.html").good());
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

// --- edge / boundary ---

void should_accept_a_null_summary() {
  ApiFixture fixture;
  EXPECT_TRUE(mlx_perf_profile_to_report(build_two_stream_graph, &fixture.payload,
                                         "c-api", "t", "build", "c_api_null_summary",
                                         nullptr) == 0);
}

void should_reject_a_callback_that_produces_no_outputs() {
  EXPECT_TRUE(mlx_perf_profile_to_report(build_nothing, nullptr, "v", "t", "build",
                                         "x", nullptr) != 0);
  EXPECT_TRUE(std::string(mlx_perf_last_error()).find("no outputs") != std::string::npos);
}

// --- negative ---

void should_surface_a_failing_callback_as_an_error_not_a_crash() {
  EXPECT_TRUE(mlx_perf_profile_to_report(build_failure, nullptr, "v", "t", "build",
                                         "x", nullptr) != 0);
  EXPECT_TRUE(std::string(mlx_perf_last_error()).find("returned 7") != std::string::npos);
}

void should_reject_null_arguments() {
  EXPECT_TRUE(mlx_perf_profile_to_report(nullptr, nullptr, "v", "t", "build", "x",
                                         nullptr) != 0);
  EXPECT_TRUE(mlx_perf_profile_to_report(build_nothing, nullptr, nullptr, "t", "build",
                                         "x", nullptr) != 0);
}

void should_report_an_unwritable_output_directory() {
  ApiFixture fixture;
  EXPECT_TRUE(mlx_perf_profile_to_report(build_two_stream_graph, &fixture.payload, "v",
                                         "t", "build/no/such/directory", "x",
                                         nullptr) != 0);
  EXPECT_TRUE(std::string(mlx_perf_last_error()).find("cannot write") !=
              std::string::npos);
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
      TEST_CASE(should_accept_a_null_summary),
      TEST_CASE(should_reject_a_callback_that_produces_no_outputs),
      TEST_CASE(should_surface_a_failing_callback_as_an_error_not_a_crash),
      TEST_CASE(should_reject_null_arguments),
      TEST_CASE(should_report_an_unwritable_output_directory),
      TEST_CASE(should_reject_a_null_mlx_c_vector),
  });
}
