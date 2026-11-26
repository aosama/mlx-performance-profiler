// Integration tests for the wired memory C entry points on the Metal GPU, as
// a non-C++ caller uses them.

#include "mlx_perf/mlx_perf_c_api.h"
#include "test_harness.h"

#include "mlx/c/array.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"

#include <cmath>
#include <string>

namespace {

bool last_error_mentions(const std::string& expected_text) {
  return std::string(mlx_perf_last_error()).find(expected_text) != std::string::npos;
}

// --- positive ---

void should_read_wired_memory_with_mlxs_holding_and_the_derived_figures() {
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  mlx_array held_array = mlx_array_new();
  const int shape[] = {1 << 24};  // 64 MiB of float32
  mlx_ones(&held_array, shape, 1, MLX_FLOAT32, gpu_stream);
  mlx_array_eval(held_array);
  mlx_perf_wired_memory reading{};
  EXPECT_TRUE(mlx_perf_read_wired_memory(&reading) == 0);
  EXPECT_TRUE(reading.system_wired_bytes > 0);
  EXPECT_TRUE(reading.wire_limit_bytes > reading.system_wired_bytes);
  EXPECT_TRUE(reading.mlx_active_bytes >= (64ull << 20));
  EXPECT_TRUE(reading.wired_outside_mlx_bytes ==
              static_cast<int64_t>(reading.system_wired_bytes) -
                  static_cast<int64_t>(reading.mlx_active_bytes + reading.mlx_cache_bytes));
  EXPECT_TRUE(reading.wire_headroom_bytes ==
              static_cast<int64_t>(reading.wire_limit_bytes) - static_cast<int64_t>(reading.system_wired_bytes));
  mlx_array_free(held_array);
  mlx_stream_free(gpu_stream);
}

// Three repetitions, at least one resolved: the wired total is system-wide,
// and another program's memory can spoil a single repetition.
void should_measure_the_wire_release_lag() {
  mlx_perf_wire_release_lag release_lag{};
  EXPECT_TRUE(mlx_perf_measure_wire_release_lag(256ull << 20, 3, &release_lag) == 0);
  EXPECT_TRUE(release_lag.allocation_bytes == (256ull << 20));
  EXPECT_TRUE(release_lag.repetition_count == 3);
  EXPECT_TRUE(release_lag.resolved_count >= 1);
  EXPECT_TRUE(release_lag.median_release_lag_us >= 0.0);
  EXPECT_TRUE(release_lag.largest_release_lag_us >= release_lag.median_release_lag_us);
  EXPECT_TRUE(release_lag.median_wired_rise_bytes > (128ll << 20));
}

// --- edge / boundary ---

void should_keep_the_wired_memory_struct_sizes_stable() {
  EXPECT_TRUE(sizeof(mlx_perf_wired_memory) == 48);
  EXPECT_TRUE(sizeof(mlx_perf_wire_release_lag) == 48);
}

// --- negative ---

void should_refuse_null_results() {
  EXPECT_TRUE(mlx_perf_read_wired_memory(nullptr) != 0);
  EXPECT_TRUE(last_error_mentions("mlx_perf_read_wired_memory: null argument"));
  EXPECT_TRUE(mlx_perf_measure_wire_release_lag(1024, 1, nullptr) != 0);
  EXPECT_TRUE(last_error_mentions("mlx_perf_measure_wire_release_lag: null argument"));
}

void should_refuse_a_probe_with_no_allocation_or_repetitions() {
  mlx_perf_wire_release_lag release_lag{};
  EXPECT_TRUE(mlx_perf_measure_wire_release_lag(0, 1, &release_lag) != 0);
  EXPECT_TRUE(last_error_mentions("allocation_bytes and repetition_count must be positive"));
  EXPECT_TRUE(mlx_perf_measure_wire_release_lag(1024, 0, &release_lag) != 0);
  EXPECT_TRUE(mlx_perf_measure_wire_release_lag(8ull << 30, 1, &release_lag) != 0);
  EXPECT_TRUE(last_error_mentions("allocation_bytes must be under 8 GiB"));
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_read_wired_memory_with_mlxs_holding_and_the_derived_figures),
      TEST_CASE(should_measure_the_wire_release_lag),
      TEST_CASE(should_keep_the_wired_memory_struct_sizes_stable),
      TEST_CASE(should_refuse_null_results),
      TEST_CASE(should_refuse_a_probe_with_no_allocation_or_repetitions),
  });
}
