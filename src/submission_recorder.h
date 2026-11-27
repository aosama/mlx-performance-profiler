#pragma once

// Submission recorder: one concern only — observe, without changing them,
// the GPU submissions and host waits MLX makes: every submission it commits
// (a Metal command buffer, a CUDA graph launch), and every time the host
// blocks waiting for the GPU.
//
// One implementation per GPU backend, chosen at configure time:
//  - Metal (metal_submission_recorder.cpp): MLX submits every command buffer
//    through MTLCommandBuffer commit (mlx/backend/metal/device.cpp); a
//    blocking eval waits in MTLSharedEvent waitUntilSignaledValue:timeoutMS:
//    (mlx/backend/metal/event.cpp); synchronize waits in MTLCommandBuffer
//    waitUntilCompleted. Objective-C runtime hooks on the concrete classes
//    MLX uses observe those three methods.
//  - CUDA (cuda_submission_recorder.cpp): MLX submits each stream's work as
//    one CUDA graph launch when it commits (cuGraphLaunch); a blocking eval
//    waits in cuEventSynchronize (mlx/backend/cuda/event.cu) and synchronize
//    in cuStreamSynchronize. MLX first checks the event (cuEventQuery), and
//    when the GPU has already finished the host goes on without blocking: a
//    check that finds the work done is the same hand-off, so it counts as a
//    wait too. CUPTI's callback API observes those calls, and its activity
//    records give each launch's GPU window (cupti_session).
//
// That shows a procedure's hand-offs between host and GPU, however many
// evaluations it makes, as the dispatch recorder shows its kernel launches.
// Each hook records only inside an open scope: on the calling thread (MLX
// submits and waits on the thread that evaluates), or on every thread, for
// a procedure that evaluates on a worker thread (#44).

#include "evaluation_boundary_listener.h"
#include "gpu_busy_time.h"
#include "recording_slots.h"

#include <memory>
#include <vector>

namespace mlx_perf {

// Installs the hooks once per process; later calls return the first result.
bool install_submission_recorder();

struct SubmissionRecord {
  // Metal command buffers or CUDA graph launches; "command buffer" in the
  // names below means either.
  int committed_command_buffer_count = 0;
  // The GPU windows of the committed submissions that completed.
  std::vector<GpuWindow> command_buffer_windows;
  // Times the host waited for the GPU (an eval's event wait or a
  // synchronize; on CUDA also a check that found the work already done),
  // and the host time spent waiting.
  int blocking_wait_count = 0;
  double blocking_wait_microseconds = 0.0;
};

// Records command buffer commits and blocking waits while alive: this
// thread's (the default), or every thread's. Scopes nest like
// DispatchRecordingScope.
class SubmissionRecordingScope {
 public:
  explicit SubmissionRecordingScope(RecordingReach recording_reach = RecordingReach::calling_thread);
  ~SubmissionRecordingScope();
  SubmissionRecordingScope(const SubmissionRecordingScope&) = delete;
  SubmissionRecordingScope& operator=(const SubmissionRecordingScope&) = delete;

  // Tells `evaluation_boundary_listener` (nullptr detaches) about every
  // blocking wait this scope records, as it returns. It must outlive the
  // scope or be detached first.
  void set_evaluation_boundary_listener(EvaluationBoundaryListener* evaluation_boundary_listener);

  // Waits for every command buffer committed so far to complete (without
  // counting that wait as the procedure's), then returns what was recorded
  // and starts a fresh record.
  SubmissionRecord finish_and_take_record();

  struct RecordingState;  // defined in the .cpp

 private:
  std::unique_ptr<RecordingState> recording_state_;
  RecordingReach recording_reach_;
  RecordingState* enclosing_recording_state_ = nullptr;
};

} // namespace mlx_perf
