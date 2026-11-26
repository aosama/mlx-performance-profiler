#include "metal_command_buffer_probe.h"

#include "metal_host_clock.h"

#include "mlx/backend/metal/device.h"

#include <algorithm>
#include <unordered_set>

namespace mlx_perf {
namespace {

constexpr double MICROSECONDS_PER_SECOND = 1e6;

// Back-dates the Metal clock to the pass anchor: sample both clocks back to
// back (sub-microsecond apart) and subtract the steady time already elapsed.
// steady_clock and mach_absolute_time tick at the same rate while the machine
// is awake; they differ only across sleep, which a millisecond pass never spans.
double anchor_host_seconds(std::chrono::steady_clock::time_point pass_started_at) {
  const double host_seconds_now = current_metal_host_seconds();
  const double seconds_since_pass_start = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - pass_started_at).count();
  return host_seconds_now - seconds_since_pass_start;
}

void release_all(std::vector<void*>& held_buffers) {
  for (void* held_buffer : held_buffers) {
    if (held_buffer != nullptr) {
      static_cast<MTL::CommandBuffer*>(held_buffer)->release();
    }
  }
}

struct BufferTimes {
  bool is_completed = false;
  double started_host_seconds = 0.0;
  double finished_host_seconds = 0.0;
};

// The unit's Event can become host-visible before Metal finishes completion
// processing for its buffer (observed: the last buffers of a pass still not
// Completed when every event had fired). Waiting here is free for accuracy —
// GPUStartTime/GPUEndTime are stamped by the GPU, not when read — and it
// only runs once the whole pass is already observed. A buffer that never
// committed or errored has no trustworthy window: reported as not completed,
// never as zero time.
BufferTimes read_buffer_times(void* held_buffer) {
  BufferTimes buffer_times;
  auto* command_buffer = static_cast<MTL::CommandBuffer*>(held_buffer);
  if (command_buffer == nullptr) {
    return buffer_times;
  }
  if (command_buffer->status() >= MTL::CommandBufferStatusCommitted) {
    command_buffer->waitUntilCompleted();
  }
  if (command_buffer->status() != MTL::CommandBufferStatusCompleted ||
      command_buffer->GPUEndTime() <= 0.0) {
    return buffer_times;
  }
  buffer_times.is_completed = true;
  buffer_times.started_host_seconds = command_buffer->GPUStartTime();
  buffer_times.finished_host_seconds = command_buffer->GPUEndTime();
  return buffer_times;
}

} // namespace

MetalCommandBufferProbe::MetalCommandBufferProbe(
    int slot_count, std::chrono::steady_clock::time_point pass_started_at)
    : position_to_command_buffer_(slot_count, nullptr),
      position_to_end_marker_buffer_(slot_count, nullptr),
      pass_anchor_host_seconds_(anchor_host_seconds(pass_started_at)) {}

MetalCommandBufferProbe::~MetalCommandBufferProbe() {
  release_all(position_to_command_buffer_);
  release_all(position_to_end_marker_buffer_);
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

void MetalCommandBufferProbe::commit_end_marker_for(int position) {
  auto* held_buffer =
      static_cast<MTL::CommandBuffer*>(position_to_command_buffer_[position]);
  if (held_buffer == nullptr || position_to_end_marker_buffer_[position] != nullptr) {
    return;
  }
  // The held buffer's own queue is MLX's queue for that stream; an empty
  // buffer changes no MLX state and encodes no work, it only takes a place
  // in line. commandBuffer() (retained references) is fine: it references
  // no resources.
  MTL::CommandBuffer* end_marker_buffer = held_buffer->commandQueue()->commandBuffer();
  end_marker_buffer->retain();
  end_marker_buffer->commit();
  position_to_end_marker_buffer_[position] = end_marker_buffer;
}

std::vector<GpuExecutionWindow> MetalCommandBufferProbe::read_windows() const {
  std::vector<GpuExecutionWindow> position_to_window(
      position_to_command_buffer_.size());
  for (std::size_t position = 0; position < position_to_window.size(); ++position) {
    const BufferTimes first_buffer_times =
        read_buffer_times(position_to_command_buffer_[position]);
    if (!first_buffer_times.is_completed) {
      continue;
    }
    GpuExecutionWindow& window = position_to_window[position];
    window.is_measured = true;
    double finished_host_seconds = first_buffer_times.finished_host_seconds;
    const BufferTimes end_marker_times =
        read_buffer_times(position_to_end_marker_buffer_[position]);
    // max(): ordering makes the marker start no earlier than the first
    // buffer's end; the max only guards against a driver that overlaps them.
    if (end_marker_times.is_completed) {
      window.is_end_bounded_by_marker = true;
      finished_host_seconds =
          std::max(finished_host_seconds, end_marker_times.started_host_seconds);
    }
    window.started_microseconds =
        (first_buffer_times.started_host_seconds - pass_anchor_host_seconds_) *
        MICROSECONDS_PER_SECOND;
    window.finished_microseconds =
        (finished_host_seconds - pass_anchor_host_seconds_) * MICROSECONDS_PER_SECOND;
  }
  return position_to_window;
}

int MetalCommandBufferProbe::count_completed_command_buffers() const {
  // A set, not a count of slots: units dispatched back to back can land in
  // the same MLX buffer when no commit separated them.
  std::unordered_set<void*> completed_buffers;
  for (void* held_buffer : position_to_command_buffer_) {
    if (read_buffer_times(held_buffer).is_completed) {
      completed_buffers.insert(held_buffer);
    }
  }
  return static_cast<int>(completed_buffers.size());
}

} // namespace mlx_perf
