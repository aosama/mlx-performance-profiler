#pragma once

// CUPTI session: one concern only — own this process's CUPTI subscription
// and hand each CUDA observer (execution probe, dispatch recorder,
// submission recorder) the driver calls and GPU activity it asked for.
//
// Why one session for all three: CUPTI accepts a single subscriber per
// process, and its activity buffers are process-wide. So the session
// subscribes once, routes each driver call to the observers registered for
// it, and keeps the GPU spans only of the launches an observer asked to
// watch; everything else CUPTI reports is dropped as it arrives, so a long
// run does not accumulate records nobody reads.
//
// Clock: the session hands CUPTI CLOCK_MONOTONIC (the clock
// std::chrono::steady_clock reads on Linux) as its timestamp source, so
// every GPU start and end it reports is already on gpu_host_clock and on
// the passes' steady_clock anchor, with no conversion.
//
// Why driver-API callbacks only: MLX reaches the GPU through both the CUDA
// runtime and the driver, and every runtime call that launches, adds a graph
// node or waits is carried out by a driver call that CUPTI also reports.
// Watching the driver alone sees each launch and wait exactly once.

#include <cupti.h>

#include <cstdint>

namespace mlx_perf {

// The span of GPU work CUPTI reported for one watched correlation id: the
// earliest start and latest end over its kernels, copies and fills.
struct GpuActivitySpan {
  bool is_measured = false;
  std::uint64_t started_nanoseconds = 0;   // CLOCK_MONOTONIC
  std::uint64_t finished_nanoseconds = 0;  // CLOCK_MONOTONIC
  int activity_record_count = 0;
};

// A driver call as an observer sees it. `function_params` points at the
// call's generated parameter struct (generated_cuda_meta.h), valid only for
// the duration of the callback.
struct DriverCall {
  CUpti_CallbackId callback_id = 0;
  bool is_entering = true;  // false on the call's return
  std::uint32_t correlation_id = 0;
  const void* function_params = nullptr;
  // On the call's return, what the driver returned; CUDA_SUCCESS on entry.
  CUresult returned_result = CUDA_SUCCESS;
};

class DriverCallObserver {
 public:
  virtual ~DriverCallObserver() = default;
  // Called on the thread that made the call. Keep it short; never call MLX
  // from it.
  virtual void on_driver_call(const DriverCall& driver_call) = 0;
};

// Starts the session once per process; later calls return the first result.
// False when CUPTI refused (no CUDA device, or another profiler such as
// Nsight Systems already holds the process's subscription).
bool start_cupti_session();

// Routes every call with `callback_id` to `observer`, which must live for the
// rest of the process (observers are process-wide singletons). At most a few
// observers per call; returns false when the session is not running or the
// table is full.
bool observe_driver_call(CUpti_CallbackId callback_id, DriverCallObserver* observer);

// Keep the GPU span of work launched under `correlation_id` (a graph launch,
// a kernel launch, a copy or a fill). Call from the launch's callback.
void watch_gpu_activity(std::uint32_t correlation_id);

// Delivers every activity record CUPTI holds, including partly filled
// buffers. Call only after the watched work is known complete.
void flush_gpu_activity();

// The span recorded for `correlation_id` since it was watched, and ends one
// watch of it (the span is dropped when its last watcher has taken it). Not
// measured when nothing was reported (never watched, work still running when
// flushed, or the launch failed).
GpuActivitySpan take_gpu_activity_span(std::uint32_t correlation_id);

// Now, on the session's clock.
std::uint64_t current_session_nanoseconds();

} // namespace mlx_perf
