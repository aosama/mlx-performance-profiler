// Integration tests for the CUDA execution probe: real graph launches on the
// GPU, checked for presence, ordering and clock alignment — never exact
// timings.

#include "cuda_execution_probe.h"
#include "test_harness.h"

#include "mlx/mlx.h"

#include <chrono>
#include <vector>

namespace {

using namespace mlx::core;
using mlx_perf::CudaExecutionProbe;
using mlx_perf::GpuExecutionWindow;
using Clock = std::chrono::steady_clock;

// The probe and the host read one clock (CLOCK_MONOTONIC), so the only slack
// is CUPTI's own timestamp conversion, well under this.
constexpr double CLOCK_SLACK_MICROSECONDS = 50.0;

double microseconds_since(Clock::time_point pass_started_at) {
  return std::chrono::duration<double, std::micro>(Clock::now() - pass_started_at).count();
}

array square_product(const Stream& gpu_stream) {
  return matmul(ones({256, 256}, float32, gpu_stream), ones({256, 256}, float32, gpu_stream), gpu_stream);
}

// --- positive ---

void should_measure_gpu_window_inside_the_host_window() {
  const Stream gpu_stream = new_stream(Device::gpu);
  array gpu_output = square_product(gpu_stream);
  eval(gpu_output.inputs());
  const Clock::time_point pass_started_at = Clock::now();
  CudaExecutionProbe execution_probe(1, pass_started_at);
  execution_probe.hold_buffer_for(0, gpu_stream);
  const double issued_microseconds = microseconds_since(pass_started_at);
  async_eval({gpu_output});
  gpu_output.wait();
  const double observed_microseconds = microseconds_since(pass_started_at);
  const std::vector<GpuExecutionWindow> windows = execution_probe.read_windows();
  EXPECT_TRUE(windows.size() == 1);
  EXPECT_TRUE(windows[0].is_measured);
  EXPECT_TRUE(!windows[0].is_end_bounded_by_marker);
  EXPECT_TRUE(windows[0].finished_microseconds >= windows[0].started_microseconds);
  EXPECT_TRUE(windows[0].started_microseconds >= issued_microseconds - CLOCK_SLACK_MICROSECONDS);
  EXPECT_TRUE(windows[0].finished_microseconds <= observed_microseconds + CLOCK_SLACK_MICROSECONDS);
  EXPECT_TRUE(execution_probe.count_completed_command_buffers() >= 1);
}

// Two units on two MLX streams evaluated by one async_eval, as a stage task
// per stream is: each stream's launch must land in its own unit.
void should_attribute_each_stream_of_one_dispatch_to_its_own_unit() {
  const Stream first_stream = new_stream(Device::gpu);
  const Stream second_stream = new_stream(Device::gpu);
  array first_output = square_product(first_stream);
  array second_output = square_product(second_stream);
  eval(first_output.inputs());
  eval(second_output.inputs());
  CudaExecutionProbe execution_probe(2, Clock::now());
  execution_probe.hold_buffer_for(0, first_stream);
  execution_probe.hold_buffer_for(1, second_stream);
  async_eval({first_output, second_output});
  execution_probe.commit_end_marker_for(0);
  execution_probe.commit_end_marker_for(1);
  first_output.wait();
  second_output.wait();
  const std::vector<GpuExecutionWindow> windows = execution_probe.read_windows();
  EXPECT_TRUE(windows[0].is_measured);
  EXPECT_TRUE(windows[1].is_measured);
  EXPECT_TRUE(execution_probe.count_completed_command_buffers() == 2);
}

// --- edge / boundary ---

// A chain long enough to make MLX commit mid-evaluation launches several
// graphs for one unit; the window must cover the last of them, which a
// single launch's window would not.
void should_span_every_graph_launch_one_unit_makes() {
  constexpr int CHAIN_LENGTH_OVER_GRAPH_BUDGET = 2000;
  const Stream gpu_stream = new_stream(Device::gpu);
  array chain = ones({1 << 16}, float32, gpu_stream);
  eval(chain);
  for (int add_index = 0; add_index < CHAIN_LENGTH_OVER_GRAPH_BUDGET; ++add_index) {
    chain = add(chain, array(1.0f), gpu_stream);
  }
  const Clock::time_point pass_started_at = Clock::now();
  CudaExecutionProbe execution_probe(1, pass_started_at);
  execution_probe.hold_buffer_for(0, gpu_stream);
  async_eval({chain});
  execution_probe.commit_end_marker_for(0);
  chain.wait();
  const double observed_microseconds = microseconds_since(pass_started_at);
  const GpuExecutionWindow window = execution_probe.read_windows()[0];
  EXPECT_TRUE(window.is_measured);
  EXPECT_TRUE(execution_probe.count_completed_command_buffers() > 1);
  EXPECT_TRUE(window.finished_microseconds <= observed_microseconds + CLOCK_SLACK_MICROSECONDS);
}

void should_leave_cpu_stream_units_unmeasured() {
  const Stream cpu_stream = new_stream(Device::cpu);
  CudaExecutionProbe execution_probe(1, Clock::now());
  execution_probe.hold_buffer_for(0, cpu_stream);
  execution_probe.commit_end_marker_for(0);
  const std::vector<GpuExecutionWindow> windows = execution_probe.read_windows();
  EXPECT_TRUE(windows.size() == 1);
  EXPECT_TRUE(!windows[0].is_measured);
}

void should_return_no_windows_for_zero_units() {
  CudaExecutionProbe execution_probe(0, Clock::now());
  EXPECT_TRUE(execution_probe.read_windows().empty());
  EXPECT_TRUE(execution_probe.count_completed_command_buffers() == 0);
}

// --- negative ---

// A held stream on which nothing was dispatched has no launch: unmeasured,
// and reading must not block.
void should_report_a_unit_without_launches_as_unmeasured() {
  const Stream gpu_stream = new_stream(Device::gpu);
  CudaExecutionProbe execution_probe(2, Clock::now());
  execution_probe.hold_buffer_for(0, gpu_stream);
  const std::vector<GpuExecutionWindow> windows = execution_probe.read_windows();
  EXPECT_TRUE(!windows[0].is_measured);
  EXPECT_TRUE(!windows[1].is_measured);
}

// Work dispatched after the unit's mark is not the unit's.
void should_not_attribute_launches_after_the_mark() {
  const Stream gpu_stream = new_stream(Device::gpu);
  array late_output = square_product(gpu_stream);
  eval(late_output.inputs());
  CudaExecutionProbe execution_probe(1, Clock::now());
  execution_probe.hold_buffer_for(0, gpu_stream);
  execution_probe.commit_end_marker_for(0);
  async_eval({late_output});
  late_output.wait();
  EXPECT_TRUE(!execution_probe.read_windows()[0].is_measured);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_measure_gpu_window_inside_the_host_window),
      TEST_CASE(should_attribute_each_stream_of_one_dispatch_to_its_own_unit),
      TEST_CASE(should_span_every_graph_launch_one_unit_makes),
      TEST_CASE(should_leave_cpu_stream_units_unmeasured),
      TEST_CASE(should_return_no_windows_for_zero_units),
      TEST_CASE(should_report_a_unit_without_launches_as_unmeasured),
      TEST_CASE(should_not_attribute_launches_after_the_mark),
  });
}
