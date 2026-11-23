#include "metal_command_buffer_probe.h"

#include "mlx/backend/metal/device.h"

#include <mach/mach_time.h>

namespace mlx_plan {
namespace {

constexpr double MICROSECONDS_PER_SECOND = 1e6;

mach_timebase_info_data_t query_timebase() {
  mach_timebase_info_data_t queried_timebase{};
  mach_timebase_info(&queried_timebase);
  return queried_timebase;
}

// The timebase Metal reports GPUStartTime/GPUEndTime in (the same one
// CACurrentMediaTime uses), computed directly to avoid a QuartzCore call.
double current_host_seconds() {
  static const mach_timebase_info_data_t timebase = query_timebase();
  const double nanoseconds = static_cast<double>(mach_absolute_time()) *
      timebase.numer / timebase.denom;
  return nanoseconds / 1e9;
}

// Back-dates the Metal clock to the pass anchor: sample both clocks back to
// back (sub-microsecond apart) and subtract the steady time already elapsed.
// steady_clock and mach_absolute_time tick at the same rate while the machine
// is awake; they differ only across sleep, which a millisecond pass never spans.
double anchor_host_seconds(std::chrono::steady_clock::time_point pass_started_at) {
  const double host_seconds_now = current_host_seconds();
  const double seconds_since_pass_start = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - pass_started_at).count();
  return host_seconds_now - seconds_since_pass_start;
}

} // namespace

MetalCommandBufferProbe::MetalCommandBufferProbe(
    int node_count, std::chrono::steady_clock::time_point pass_started_at)
    : position_to_command_buffer_(node_count, nullptr),
      pass_anchor_host_seconds_(anchor_host_seconds(pass_started_at)) {}

MetalCommandBufferProbe::~MetalCommandBufferProbe() {
  for (void* held_buffer : position_to_command_buffer_) {
    if (held_buffer != nullptr) {
      static_cast<MTL::CommandBuffer*>(held_buffer)->release();
    }
  }
}

void MetalCommandBufferProbe::hold_buffer_for(
    int position, const mlx::core::Stream& stream) {
  if (stream.device != mlx::core::Device::gpu) {
    return;
  }
  MTL::CommandBuffer* upcoming_buffer =
      mlx::core::metal::get_command_encoder(stream).get_command_buffer();
  // Retained because MLX drops its own reference on commit; the timestamps
  // are read only after the GPU has finished with the buffer. A re-hold of
  // the same position releases the earlier buffer so nothing leaks.
  upcoming_buffer->retain();
  if (position_to_command_buffer_[position] != nullptr) {
    static_cast<MTL::CommandBuffer*>(position_to_command_buffer_[position])
        ->release();
  }
  position_to_command_buffer_[position] = upcoming_buffer;
}

std::vector<GpuExecutionWindow> MetalCommandBufferProbe::read_windows() const {
  std::vector<GpuExecutionWindow> position_to_window(
      position_to_command_buffer_.size());
  for (std::size_t position = 0; position < position_to_window.size();
       ++position) {
    auto* held_buffer =
        static_cast<MTL::CommandBuffer*>(position_to_command_buffer_[position]);
    if (held_buffer == nullptr) {
      continue;
    }
    // The node's Event can become host-visible before Metal finishes
    // completion processing for its buffer (observed: the last buffers of a
    // pass still not Completed when every event had fired). Waiting here is
    // free for accuracy — GPUStartTime/GPUEndTime are stamped by the GPU, not
    // when read — and it only runs once the whole pass is already observed.
    if (held_buffer->status() >= MTL::CommandBufferStatusCommitted) {
      held_buffer->waitUntilCompleted();
    }
    // A buffer that never committed or errored has no trustworthy window; it
    // is reported as unmeasured, never as zero time.
    if (held_buffer->status() != MTL::CommandBufferStatusCompleted ||
        held_buffer->GPUEndTime() <= 0.0) {
      continue;
    }
    GpuExecutionWindow& window = position_to_window[position];
    window.is_measured = true;
    window.started_microseconds =
        (held_buffer->GPUStartTime() - pass_anchor_host_seconds_) *
        MICROSECONDS_PER_SECOND;
    window.finished_microseconds =
        (held_buffer->GPUEndTime() - pass_anchor_host_seconds_) *
        MICROSECONDS_PER_SECOND;
  }
  return position_to_window;
}

} // namespace mlx_plan
