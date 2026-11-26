#pragma once

// Metal dispatch recorder: one concern only — observe the kernel launches
// MLX encodes, without changing them: each launch's grid and threadgroup
// shape, its pipeline's limits and threadgroup memory, and the Metal
// function the pipeline was compiled from.
//
// Why at the Metal level: MLX encodes every launch through
// CommandEncoder::dispatch_threadgroups / dispatch_threads
// (mlx/backend/metal/device.cpp), which are not virtual and drive a private
// Metal encoder. So the recorder installs Objective-C runtime hooks on the
// concrete classes of the Metal compute encoder and device MLX uses. Each
// hook calls the original implementation first and only then records, and
// it records only while a DispatchRecordingScope is open on the calling
// thread. MLX encodes GPU work inline on the thread that evaluates
// (transforms.cpp: gpu::eval), so a scope around one evaluation sees
// exactly that evaluation's launches, and other threads' Metal work is never
// recorded. Outside a scope a hook costs one thread-local pointer check, so
// the timed passes are not slowed.

#include "dispatch_occupancy.h"

#include <memory>
#include <vector>

namespace mlx_perf {

// Installs the hooks once per process; later calls return the first result.
// False when they could not be installed (no Metal GPU, or the encoder class
// lacks the expected methods): scopes then record nothing.
bool install_dispatch_recorder();

// Records every kernel launch MLX encodes on this thread while it is alive.
// Scopes nest: an inner scope records into itself, and the outer one
// resumes when it ends.
class DispatchRecordingScope {
 public:
  DispatchRecordingScope();
  ~DispatchRecordingScope();
  DispatchRecordingScope(const DispatchRecordingScope&) = delete;
  DispatchRecordingScope& operator=(const DispatchRecordingScope&) = delete;

  // The launches recorded so far, in encoding order; the scope keeps
  // recording afterwards into an empty list.
  std::vector<KernelDispatch> take_recorded_dispatches();

  // The recording state the hooks write into; defined in the .cpp.
  struct RecordingState;

 private:
  std::unique_ptr<RecordingState> recording_state_;
  RecordingState* enclosing_recording_state_ = nullptr;
};

} // namespace mlx_perf
