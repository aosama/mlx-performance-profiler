// Integration tests for the Metal command-buffer probe: real GPU buffers,
// checked for presence, ordering and clock alignment — never exact timings.

#include "metal_command_buffer_probe.h"
#include "test_harness.h"

#include "mlx/mlx.h"

#include <chrono>
#include <vector>

namespace {

using namespace mlx::core;
using mlx_perf::GpuExecutionWindow;
using mlx_perf::MetalCommandBufferProbe;
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

// A chain far over MLX's per-buffer op budget commits mid-eval, so the held
// first buffer ends long before the work does; the end marker must extend
// the window to the last buffer.
void should_extend_the_window_to_the_last_buffer_with_an_end_marker() {
  constexpr int CHAIN_LENGTH_OVER_BUFFER_BUDGET = 300;
  const Stream gpu_stream = new_stream(Device::gpu);
  array chain = ones({1 << 20}, float32, gpu_stream);
  eval(chain);
  for (int add_index = 0; add_index < CHAIN_LENGTH_OVER_BUFFER_BUDGET; ++add_index) {
    chain = add(chain, array(1.0f), gpu_stream);
  }
  const Clock::time_point pass_started_at = Clock::now();
  MetalCommandBufferProbe first_buffer_only(1, pass_started_at);
  MetalCommandBufferProbe with_end_marker(1, pass_started_at);
  first_buffer_only.hold_buffer_for(0, gpu_stream);
  with_end_marker.hold_buffer_for(0, gpu_stream);
  async_eval({chain});
  with_end_marker.commit_end_marker_for(0);
  chain.wait();
  const GpuExecutionWindow first_window = first_buffer_only.read_windows()[0];
  const GpuExecutionWindow marked_window = with_end_marker.read_windows()[0];
  EXPECT_TRUE(first_window.is_measured && !first_window.is_end_bounded_by_marker);
  EXPECT_TRUE(marked_window.is_measured && marked_window.is_end_bounded_by_marker);
  // Same first buffer, so the same start up to the two probes' independent
  // clock-anchor samples (each takes two back-to-back clock reads).
  constexpr double ANCHOR_JITTER_MICROSECONDS = 5.0;
  EXPECT_NEAR(marked_window.started_microseconds, first_window.started_microseconds,
              ANCHOR_JITTER_MICROSECONDS);
  // The marked end is strictly later than the first buffer's end.
  EXPECT_TRUE(marked_window.finished_microseconds > first_window.finished_microseconds);
}

// --- edge / boundary ---

void should_ignore_an_end_marker_on_a_cpu_stream() {
  const Stream cpu_stream = new_stream(Device::cpu);
  MetalCommandBufferProbe command_buffer_probe(1, Clock::now());
  command_buffer_probe.hold_buffer_for(0, cpu_stream);
  command_buffer_probe.commit_end_marker_for(0);
  const GpuExecutionWindow window = command_buffer_probe.read_windows()[0];
  EXPECT_TRUE(!window.is_measured && !window.is_end_bounded_by_marker);
}

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

// A marker without a held buffer has no queue to join; it must be a no-op,
// not a crash, and must not invent a window.
void should_not_commit_an_end_marker_without_a_held_buffer() {
  MetalCommandBufferProbe command_buffer_probe(1, Clock::now());
  command_buffer_probe.commit_end_marker_for(0);
  const GpuExecutionWindow window = command_buffer_probe.read_windows()[0];
  EXPECT_TRUE(!window.is_measured && !window.is_end_bounded_by_marker);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_measure_gpu_window_aligned_to_the_host_clock),
      TEST_CASE(should_leave_cpu_stream_nodes_unmeasured),
      TEST_CASE(should_return_no_windows_for_zero_nodes),
      TEST_CASE(should_report_uncommitted_buffer_as_unmeasured_without_blocking),
      TEST_CASE(should_extend_the_window_to_the_last_buffer_with_an_end_marker),
      TEST_CASE(should_ignore_an_end_marker_on_a_cpu_stream),
      TEST_CASE(should_not_commit_an_end_marker_without_a_held_buffer),
  });
}
