#include "submission_recorder.h"

#include "cuda_gpu_synchronization.h"
#include "cuda_host_hand_off.h"
#include "cupti_session.h"
#include "gpu_host_clock.h"
#include "recording_slots.h"

#include <cuda.h>

#include <cstdint>

namespace mlx_perf {

struct SubmissionRecordingScope::RecordingState {
  // Each graph launch's CUPTI correlation id, watched in cupti_session until
  // its GPU span is read.
  std::vector<std::uint32_t> launch_correlation_ids;
  int blocking_wait_count = 0;
  double blocking_wait_microseconds = 0.0;
  EvaluationBoundaryListener* evaluation_boundary_listener = nullptr;
};

namespace {

using RecordingState = SubmissionRecordingScope::RecordingState;
using SubmissionSlots = RecordingSlots<RecordingState>;
using ActiveRecording = SubmissionSlots::ActiveRecording;

// When the calling thread's current wait began. A thread waits on one thing
// at a time, so one slot per thread is enough.
thread_local double wait_started_seconds = 0.0;

void record_graph_launch(RecordingState& recording_state, const DriverCall& driver_call) {
  recording_state.launch_correlation_ids.push_back(driver_call.correlation_id);
  watch_gpu_activity(driver_call.correlation_id);
}

// Looked up after the wait, never held across it: a process-wide recording's
// lock must not be held while a thread blocks on the GPU. A wait counts when
// a recording is open as it ends, as on Metal.
void count_blocking_wait() {
  const double wait_finished_seconds = current_gpu_host_seconds();
  const ActiveRecording active_recording;
  if (active_recording.recording_state() == nullptr) {
    return;
  }
  RecordingState& recording_state = *active_recording.recording_state();
  ++recording_state.blocking_wait_count;
  recording_state.blocking_wait_microseconds += (wait_finished_seconds - wait_started_seconds) * 1e6;
  // Under the recording's lock, so the listener cannot be detached while it
  // runs; it only reads counters (see evaluation_boundary_listener.h).
  if (recording_state.evaluation_boundary_listener != nullptr) {
    recording_state.evaluation_boundary_listener->on_evaluation_boundary();
  }
}

class SubmissionObserver final : public DriverCallObserver {
 public:
  void on_driver_call(const DriverCall& driver_call) override {
    if (is_graph_launch(driver_call.callback_id)) {
      if (driver_call.is_entering) {
        const ActiveRecording active_recording;
        if (active_recording.recording_state() != nullptr) {
          record_graph_launch(*active_recording.recording_state(), driver_call);
        }
      }
      return;
    }
    // Every other observed call is a wait or a completion check. The clock
    // is read around each, recording or not: two clock reads, negligible
    // beside a wait for the GPU.
    if (driver_call.is_entering) {
      wait_started_seconds = current_gpu_host_seconds();
    } else if (counts_as_host_hand_off(driver_call.callback_id, driver_call.returned_result)) {
      count_blocking_wait();
    }
  }
};

// MLX commits each stream's work as one graph launch (cu::CommandEncoder::
// commit); a blocking eval waits on a CUDA event (cu::CudaEvent::wait,
// cudaEventSynchronize) and synchronize on the stream (cudaStreamSynchronize).
// The context-wide wait is included for callers that synchronize the device,
// and the event check MLX makes before it waits (cuda_host_hand_off).
constexpr CUpti_CallbackId OBSERVED_SUBMISSION_CALLS[] = {
    CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch,           CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuEventSynchronize,      CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize,
    CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize_ptsz, CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize,
    CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize_v2,     CUPTI_DRIVER_TRACE_CBID_cuEventQuery,
};

bool install_submission_observer() {
  static SubmissionObserver submission_observer;
  bool is_installed = true;
  for (const CUpti_CallbackId submission_call : OBSERVED_SUBMISSION_CALLS) {
    is_installed &= observe_driver_call(submission_call, &submission_observer);
  }
  return is_installed;
}

// Seconds on gpu_host_clock, which is the session's clock in nanoseconds.
GpuWindow to_gpu_window(const GpuActivitySpan& activity_span) {
  return GpuWindow{static_cast<double>(activity_span.started_nanoseconds) / 1e9,
                   static_cast<double>(activity_span.finished_nanoseconds) / 1e9};
}

} // namespace

bool install_submission_recorder() {
  static const bool is_installed = install_submission_observer();
  return is_installed;
}

SubmissionRecordingScope::SubmissionRecordingScope(RecordingReach recording_reach)
    : recording_state_(std::make_unique<RecordingState>()), recording_reach_(recording_reach) {
  enclosing_recording_state_ = SubmissionSlots::swap_open_recording(recording_reach_, recording_state_.get());
}

// Closed first, so no hook on another thread can add a launch while the
// watched ones are given up.
SubmissionRecordingScope::~SubmissionRecordingScope() {
  SubmissionSlots::swap_open_recording(recording_reach_, enclosing_recording_state_);
  for (const std::uint32_t correlation_id : recording_state_->launch_correlation_ids) {
    take_gpu_activity_span(correlation_id);
  }
}

// Paused while attached, like a take, so no hook reads the pointer mid-write.
void SubmissionRecordingScope::set_evaluation_boundary_listener(
    EvaluationBoundaryListener* evaluation_boundary_listener) {
  SubmissionSlots::swap_open_recording(recording_reach_, nullptr);
  recording_state_->evaluation_boundary_listener = evaluation_boundary_listener;
  SubmissionSlots::swap_open_recording(recording_reach_, recording_state_.get());
}

SubmissionRecord SubmissionRecordingScope::finish_and_take_record() {
  // The profiler's own waits below are not the procedure's: recording
  // pauses while they run (and, process-wide, while the record is read).
  SubmissionSlots::swap_open_recording(recording_reach_, nullptr);
  wait_for_every_gpu_submission();
  flush_gpu_activity();
  SubmissionRecord submission_record;
  submission_record.committed_command_buffer_count =
      static_cast<int>(recording_state_->launch_correlation_ids.size());
  for (const std::uint32_t correlation_id : recording_state_->launch_correlation_ids) {
    const GpuActivitySpan activity_span = take_gpu_activity_span(correlation_id);
    if (activity_span.is_measured) {
      submission_record.command_buffer_windows.push_back(to_gpu_window(activity_span));
    }
  }
  submission_record.blocking_wait_count = recording_state_->blocking_wait_count;
  submission_record.blocking_wait_microseconds = recording_state_->blocking_wait_microseconds;
  EvaluationBoundaryListener* evaluation_boundary_listener = recording_state_->evaluation_boundary_listener;
  *recording_state_ = RecordingState{};
  recording_state_->evaluation_boundary_listener = evaluation_boundary_listener;
  SubmissionSlots::swap_open_recording(recording_reach_, recording_state_.get());
  return submission_record;
}

} // namespace mlx_perf
