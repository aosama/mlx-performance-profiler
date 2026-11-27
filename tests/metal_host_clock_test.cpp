// Integration tests for metal_host_clock on the Metal GPU: the clock must be
// the one Metal stamps command buffers with, or GPU windows and host instants
// cannot be compared. Checked for ordering and alignment, never exact timings.

#include "gpu_host_clock.h"
#include "test_harness.h"

#include "mlx/backend/metal/device.h"
#include "mlx/mlx.h"

#include <chrono>
#include <thread>

namespace {

using mlx_perf::current_gpu_host_seconds;

// --- positive ---

void should_bracket_a_command_buffers_gpu_window() {
  MTL::Device* metal_device = mlx::core::metal::device(mlx::core::Device::gpu).mtl_device();
  NS::AutoreleasePool* autorelease_pool = NS::AutoreleasePool::alloc()->init();
  MTL::CommandQueue* probe_queue = metal_device->newCommandQueue();
  MTL::CommandBuffer* probe_buffer = probe_queue->commandBuffer();
  const double before_commit_seconds = current_gpu_host_seconds();
  probe_buffer->commit();
  probe_buffer->waitUntilCompleted();
  const double after_completion_seconds = current_gpu_host_seconds();
  EXPECT_TRUE(probe_buffer->GPUStartTime() >= before_commit_seconds);
  EXPECT_TRUE(probe_buffer->GPUEndTime() <= after_completion_seconds);
  probe_queue->release();
  autorelease_pool->release();
}

void should_advance_with_wall_time() {
  const double started_seconds = current_gpu_host_seconds();
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  const double elapsed_seconds = current_gpu_host_seconds() - started_seconds;
  EXPECT_TRUE(elapsed_seconds >= 0.015 && elapsed_seconds < 1.0);
}

// --- edge / boundary ---

void should_never_run_backwards_between_immediate_reads() {
  double previous_seconds = current_gpu_host_seconds();
  for (int read_index = 0; read_index < 1000; ++read_index) {
    const double current_seconds = current_gpu_host_seconds();
    EXPECT_TRUE(current_seconds >= previous_seconds);
    previous_seconds = current_seconds;
  }
}

// --- negative ---

void should_not_read_zero_or_a_negative_time() {
  // Zero is what Metal reports for a buffer that never ran: a clock reading
  // of zero would be indistinguishable from it.
  EXPECT_TRUE(current_gpu_host_seconds() > 0.0);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_bracket_a_command_buffers_gpu_window),
      TEST_CASE(should_advance_with_wall_time),
      TEST_CASE(should_never_run_backwards_between_immediate_reads),
      TEST_CASE(should_not_read_zero_or_a_negative_time),
  });
}
