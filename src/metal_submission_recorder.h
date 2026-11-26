#pragma once

// Metal submission recorder: one concern only — observe, without changing
// them, the GPU submissions and host waits MLX makes: every command buffer it
// commits, and every time the host blocks waiting for the GPU.
//
// Why these three methods: MLX submits every command buffer through
// MTLCommandBuffer commit (mlx/backend/metal/device.cpp); a blocking eval
// waits in MTLSharedEvent waitUntilSignaledValue:timeoutMS:
// (mlx/backend/metal/event.cpp); synchronize waits in MTLCommandBuffer
// waitUntilCompleted. Hooking those on the concrete classes MLX uses shows a
// procedure's hand-offs between host and GPU, however many evaluations it
// makes, as metal_dispatch_recorder shows its kernel launches. Each hook
// calls the original first and records only inside an open scope: on the
// calling thread (MLX submits and waits on the thread that evaluates), or on
// every thread, for a procedure that evaluates on a worker thread (#44).

#include "evaluation_boundary_listener.h"
#include "gpu_busy_time.h"
#include "recording_slots.h"

#include <memory>
#include <vector>

namespace mlx_perf {

// Installs the hooks once per process; later calls return the first result.
bool install_submission_recorder();

struct SubmissionRecord {
  int committed_command_buffer_count = 0;
  // The GPU windows of the committed command buffers that completed.
  std::vector<GpuWindow> command_buffer_windows;
  // Times the host blocked waiting for the GPU (an eval's event wait or a
  // synchronize), and the host time spent blocked.
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
