#pragma once

// Dispatch recorder: one concern only — observe the kernel launches MLX
// encodes, without changing them: each launch's grid and block
// (threadgroup) shape, its kernel's limits and shared (threadgroup) memory,
// and the kernel's function name.
//
// One implementation per GPU backend, chosen at configure time:
//  - Metal (metal_dispatch_recorder.cpp): MLX encodes every launch through
//    CommandEncoder::dispatch_threadgroups / dispatch_threads
//    (mlx/backend/metal/device.cpp), which are not virtual and drive a
//    private Metal encoder, so the recorder installs Objective-C runtime
//    hooks on the concrete classes of the Metal compute encoder and device
//    MLX uses. Each hook calls the original implementation first and only
//    then records.
//  - CUDA (cuda_dispatch_recorder.cpp): MLX adds every kernel to a CUDA
//    graph (cuGraphAddKernelNode) or, with graphs off and for cuBLAS and
//    cuDNN work it captures, launches it (cuLaunchKernel and
//    cuLaunchKernelEx). The recorder subscribes to those driver calls
//    through CUPTI's callback API (cupti_session), NVIDIA's observation
//    interface, and reads the launch shape from the call's parameters.
//
// Either way it records only while a DispatchRecordingScope is open. MLX
// encodes GPU work inline on the thread that evaluates (transforms.cpp:
// gpu::eval), so a calling-thread scope around one evaluation sees exactly
// that evaluation's launches and never other threads' GPU work; a
// whole-process scope (for a procedure that evaluates on a worker thread)
// sees every thread's. Outside a scope a hook costs a thread-local and an
// atomic check (recording_slots), so the timed passes are not slowed.

#include "dispatch_occupancy.h"
#include "recording_slots.h"

#include <memory>
#include <vector>

namespace mlx_perf {

// Installs the hooks once per process; later calls return the first result.
// False when they could not be installed (no GPU of the backend, or, on
// Metal, an encoder class without the expected methods): scopes then record
// nothing.
bool install_dispatch_recorder();

// Records every kernel launch MLX encodes while it is alive: on this thread
// (the default), or on any thread. Scopes of one reach nest: an inner scope
// records into itself, and the outer one resumes when it ends. A thread's own
// scope wins over a whole-process one.
class DispatchRecordingScope {
 public:
  explicit DispatchRecordingScope(RecordingReach recording_reach = RecordingReach::calling_thread);
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
  RecordingReach recording_reach_;
  RecordingState* enclosing_recording_state_ = nullptr;
};

} // namespace mlx_perf
