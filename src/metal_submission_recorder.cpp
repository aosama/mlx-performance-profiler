#include "metal_submission_recorder.h"

#include "metal_host_clock.h"
#include "metal_object_classes.h"
#include "objc_method_hooks.h"
#include "recording_slots.h"

#include "mlx/backend/metal/device.h"

#include <cstdint>

namespace mlx_perf {

struct SubmissionRecordingScope::RecordingState {
  // Retained until read: MLX releases a buffer once it completes, and its
  // GPU window is read after that.
  std::vector<MTL::CommandBuffer*> committed_buffers;
  int blocking_wait_count = 0;
  double blocking_wait_microseconds = 0.0;
};

namespace {

using RecordingState = SubmissionRecordingScope::RecordingState;

using SubmissionSlots = RecordingSlots<RecordingState>;
using ActiveRecording = SubmissionSlots::ActiveRecording;

using CommitImp = void (*)(id, SEL);
using WaitUntilCompletedImp = void (*)(id, SEL);
using WaitUntilSignaledValueImp = bool (*)(id, SEL, std::uint64_t, std::uint64_t);

CommitImp original_commit = nullptr;
WaitUntilCompletedImp original_wait_until_completed = nullptr;
WaitUntilSignaledValueImp original_wait_until_signaled_value = nullptr;

// Looked up after the wait, never held across it: a process-wide recording's
// lock must not be held while a thread blocks on the GPU. A wait counts when
// a recording is open as it ends.
void count_blocking_wait(double wait_started_seconds) {
  const double wait_finished_seconds = current_metal_host_seconds();
  const ActiveRecording active_recording;
  if (active_recording.recording_state() != nullptr) {
    ++active_recording.recording_state()->blocking_wait_count;
    active_recording.recording_state()->blocking_wait_microseconds +=
        (wait_finished_seconds - wait_started_seconds) * 1e6;
  }
}

void commit_hook(id command_buffer_object, SEL selector) {
  original_commit(command_buffer_object, selector);
  const ActiveRecording active_recording;
  if (active_recording.recording_state() != nullptr) {
    auto* command_buffer = reinterpret_cast<MTL::CommandBuffer*>(command_buffer_object);
    command_buffer->retain();
    active_recording.recording_state()->committed_buffers.push_back(command_buffer);
  }
}

// The clock is read around every wait, recording or not: two reads of
// mach_absolute_time, negligible beside a wait for the GPU.
void wait_until_completed_hook(id command_buffer_object, SEL selector) {
  const double wait_started_seconds = current_metal_host_seconds();
  original_wait_until_completed(command_buffer_object, selector);
  count_blocking_wait(wait_started_seconds);
}

bool wait_until_signaled_value_hook(id shared_event_object, SEL selector, std::uint64_t awaited_value,
                                    std::uint64_t timeout_milliseconds) {
  const double wait_started_seconds = current_metal_host_seconds();
  const bool is_signaled =
      original_wait_until_signaled_value(shared_event_object, selector, awaited_value, timeout_milliseconds);
  count_blocking_wait(wait_started_seconds);
  return is_signaled;
}

bool install_hooks() {
  const MetalObjectClasses& metal_object_classes = find_metal_object_classes();
  if (metal_object_classes.command_buffer_class == nullptr || metal_object_classes.shared_event_class == nullptr) {
    return false;
  }
  bool is_installed = replace_method(metal_object_classes.command_buffer_class, "commit",
                                     reinterpret_cast<IMP>(commit_hook), original_commit);
  is_installed &= replace_method(metal_object_classes.command_buffer_class, "waitUntilCompleted",
                                 reinterpret_cast<IMP>(wait_until_completed_hook), original_wait_until_completed);
  is_installed &= replace_method(metal_object_classes.shared_event_class, "waitUntilSignaledValue:timeoutMS:",
                                 reinterpret_cast<IMP>(wait_until_signaled_value_hook),
                                 original_wait_until_signaled_value);
  return is_installed;
}

// A completed buffer's GPU window; buffers that errored or never ran have no
// trustworthy window and are left out.
void read_window(MTL::CommandBuffer* command_buffer, std::vector<GpuWindow>& command_buffer_windows) {
  if (command_buffer->status() != MTL::CommandBufferStatusCompleted) {
    return;
  }
  const double started_seconds = command_buffer->GPUStartTime();
  const double finished_seconds = command_buffer->GPUEndTime();
  if (started_seconds > 0.0 && finished_seconds >= started_seconds) {
    command_buffer_windows.push_back(GpuWindow{started_seconds, finished_seconds});
  }
}

} // namespace

bool install_submission_recorder() {
  static const bool is_installed = install_hooks();
  return is_installed;
}

SubmissionRecordingScope::SubmissionRecordingScope(RecordingReach recording_reach)
    : recording_state_(std::make_unique<RecordingState>()), recording_reach_(recording_reach) {
  enclosing_recording_state_ = SubmissionSlots::swap_open_recording(recording_reach_, recording_state_.get());
}

// Closed first, so no hook on another thread can add a buffer while the
// retained ones are released.
SubmissionRecordingScope::~SubmissionRecordingScope() {
  SubmissionSlots::swap_open_recording(recording_reach_, enclosing_recording_state_);
  for (MTL::CommandBuffer* committed_buffer : recording_state_->committed_buffers) {
    committed_buffer->release();
  }
}

SubmissionRecord SubmissionRecordingScope::finish_and_take_record() {
  // The profiler's own waits below are not the procedure's: recording
  // pauses while they run (and, process-wide, while the record is read).
  SubmissionSlots::swap_open_recording(recording_reach_, nullptr);
  SubmissionRecord submission_record;
  submission_record.committed_command_buffer_count =
      static_cast<int>(recording_state_->committed_buffers.size());
  for (MTL::CommandBuffer* committed_buffer : recording_state_->committed_buffers) {
    committed_buffer->waitUntilCompleted();
    read_window(committed_buffer, submission_record.command_buffer_windows);
    committed_buffer->release();
  }
  submission_record.blocking_wait_count = recording_state_->blocking_wait_count;
  submission_record.blocking_wait_microseconds = recording_state_->blocking_wait_microseconds;
  *recording_state_ = RecordingState{};
  SubmissionSlots::swap_open_recording(recording_reach_, recording_state_.get());
  return submission_record;
}

} // namespace mlx_perf
