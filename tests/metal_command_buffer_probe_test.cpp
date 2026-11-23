// Integration tests for the Metal command-buffer probe: real GPU buffers,
// checked for presence, ordering and clock alignment — never exact timings.

#include "metal_command_buffer_probe.h"
#include "test_harness.h"

#include "mlx/mlx.h"

#include <chrono>
#include <vector>

namespace {

using namespace mlx::core;
using mlx_plan::GpuExecutionWindow;
using mlx_plan::MetalCommandBufferProbe;
using Clock = std::chrono::steady_clock;

double microseconds_since(Clock::time_point pass_started_at) {
  return std::chrono::duration<double, std::micro>(
             Clock::now() - pass_started_at).count();
}

// --- positive ---

void should_measure_gpu_window_aligned_to_the_host_clock() {
  const Stream gpu_stream = new_stream(Device::gpu);
  array gpu_output = matmul(
      ones({256, 256}, float32, gpu_stream),
      ones({256, 256}, float32, gpu_stream), gpu_stream);
  eval(gpu_output.inputs());
  const Clock::time_point pass_started_at = Clock::now();
  MetalCommandBufferProbe command_buffer_probe(1, pass_started_at);
  command_buffer_probe.hold_buffer_for(0, gpu_stream);
  const double issued_microseconds = microseconds_since(pass_started_at);
  async_eval({gpu_output});
  gpu_output.wait();
  const double observed_microseconds = microseconds_since(pass_started_at);
  const std::vector<GpuExecutionWindow> windows =
      command_buffer_probe.read_windows();
  EXPECT_TRUE(windows.size() == 1);
  EXPECT_TRUE(windows[0].is_measured);
  EXPECT_TRUE(windows[0].finished_microseconds >= windows[0].started_microseconds);
  // Clock alignment: the GPU cannot start before the host issued the work,
  // nor finish after the host observed it finished (small anchor slack).
  constexpr double ANCHOR_SLACK_MICROSECONDS = 50.0;
  EXPECT_TRUE(windows[0].started_microseconds >=
              issued_microseconds - ANCHOR_SLACK_MICROSECONDS);
  EXPECT_TRUE(windows[0].finished_microseconds <=
              observed_microseconds + ANCHOR_SLACK_MICROSECONDS);
}

// --- edge / boundary ---

void should_leave_cpu_stream_nodes_unmeasured() {
  const Stream cpu_stream = new_stream(Device::cpu);
  MetalCommandBufferProbe command_buffer_probe(1, Clock::now());
  command_buffer_probe.hold_buffer_for(0, cpu_stream);
  const std::vector<GpuExecutionWindow> windows =
      command_buffer_probe.read_windows();
  EXPECT_TRUE(windows.size() == 1);
  EXPECT_TRUE(!windows[0].is_measured);
}

void should_return_no_windows_for_zero_nodes() {
  MetalCommandBufferProbe command_buffer_probe(0, Clock::now());
  EXPECT_TRUE(command_buffer_probe.read_windows().empty());
}

// --- negative ---

// A held buffer that was never committed (nothing dispatched) must read as
// unmeasured — and reading must not block waiting for it.
void should_report_uncommitted_buffer_as_unmeasured_without_blocking() {
  const Stream gpu_stream = new_stream(Device::gpu);
  MetalCommandBufferProbe command_buffer_probe(2, Clock::now());
  command_buffer_probe.hold_buffer_for(0, gpu_stream);
  const std::vector<GpuExecutionWindow> windows =
      command_buffer_probe.read_windows();
  EXPECT_TRUE(!windows[0].is_measured);
  EXPECT_TRUE(!windows[1].is_measured);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_measure_gpu_window_aligned_to_the_host_clock),
      TEST_CASE(should_leave_cpu_stream_nodes_unmeasured),
      TEST_CASE(should_return_no_windows_for_zero_nodes),
      TEST_CASE(should_report_uncommitted_buffer_as_unmeasured_without_blocking),
  });
}
