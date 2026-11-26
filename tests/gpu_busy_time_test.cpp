// Unit tests for gpu_busy_time: how long the GPU was busy inside a span,
// counting overlapping command buffers once. Pure arithmetic, no GPU.

#include "gpu_busy_time.h"
#include "test_harness.h"

#include <cmath>

namespace {

using mlx_perf::GpuWindow;
using mlx_perf::measure_gpu_busy_microseconds;

bool is_close(double measured_microseconds, double expected_microseconds) {
  return std::abs(measured_microseconds - expected_microseconds) < 1e-6;
}

// --- positive ---

void should_add_disjoint_windows() {
  EXPECT_TRUE(is_close(measure_gpu_busy_microseconds({{1.0, 1.000010}, {1.000020, 1.000025}}, 1.0, 2.0), 15.0));
}

void should_count_overlapping_windows_on_two_streams_once() {
  // 10 µs and 10 µs overlapping by 5 µs: 15 µs of GPU busy, not 20.
  EXPECT_TRUE(is_close(measure_gpu_busy_microseconds({{1.000005, 1.000015}, {1.0, 1.000010}}, 1.0, 2.0), 15.0));
}

// --- edge / boundary ---

void should_clip_windows_to_the_span() {
  EXPECT_TRUE(is_close(measure_gpu_busy_microseconds({{0.999990, 1.000010}}, 1.0, 1.000005), 5.0));
}

void should_count_a_window_nested_inside_another_once() {
  EXPECT_TRUE(is_close(measure_gpu_busy_microseconds({{1.0, 1.000020}, {1.000005, 1.000010}}, 1.0, 2.0), 20.0));
}

void should_measure_nothing_without_windows() {
  EXPECT_TRUE(measure_gpu_busy_microseconds({}, 1.0, 2.0) == 0.0);
}

// --- negative ---

void should_measure_nothing_for_an_inverted_span_or_window() {
  EXPECT_TRUE(measure_gpu_busy_microseconds({{1.0, 1.5}}, 2.0, 1.0) == 0.0);
  EXPECT_TRUE(measure_gpu_busy_microseconds({{1.5, 1.2}}, 1.0, 2.0) == 0.0);
}

void should_ignore_windows_entirely_outside_the_span() {
  EXPECT_TRUE(measure_gpu_busy_microseconds({{0.5, 0.6}, {2.5, 2.6}}, 1.0, 2.0) == 0.0);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_add_disjoint_windows),
      TEST_CASE(should_count_overlapping_windows_on_two_streams_once),
      TEST_CASE(should_clip_windows_to_the_span),
      TEST_CASE(should_count_a_window_nested_inside_another_once),
      TEST_CASE(should_measure_nothing_without_windows),
      TEST_CASE(should_measure_nothing_for_an_inverted_span_or_window),
      TEST_CASE(should_ignore_windows_entirely_outside_the_span),
  });
}
