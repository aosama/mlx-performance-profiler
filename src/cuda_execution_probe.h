#pragma once

// CUDA execution probe: one concern only — read the GPU's own execution
// window for each timed unit (a node, or a stage task), on the pass clock.
//
// Why this works without touching MLX (0.32.5, backend/cuda/device.cpp):
// every async_eval ends in gpu::finalize, which commits each used MLX
// stream's CUDA graph with one cuGraphLaunch on that stream's CUDA stream,
// on the thread that evaluates. So the launches made on a held stream
// between hold_buffer_for and the next hold (or commit_end_marker_for) are
// exactly that unit's work. The probe notes each one's CUPTI correlation id
// as it is launched, and CUPTI's activity records then give the start and
// end of every kernel, copy and fill the launch ran, already on the pass's
// clock (cupti_session).
//
// What it improves on the Metal probe: the window is the span of the unit's
// own GPU work, across every graph launch the unit made, so no end marker
// is needed (is_end_bounded_by_marker stays false) and a cross-stream wait
// encoded at the head of a graph is not counted as compute.
//
// Honest limits:
//  - Only graph launches are attributed. With MLX_USE_CUDA_GRAPHS=0, MLX
//    launches kernels directly and the probe reports no window.
//  - CUPTI may start after MLX has instantiated and cached a graph: its
//    kernels are still traced when MLX relaunches it (measured on MLX 0.32.5
//    with the CUDA 13 driver), so no observer has to start before the job.
//  - CPU-device streams have no CUDA stream; they report no window.

#include "gpu_execution_window.h"

#include "mlx/stream.h"

#include <chrono>
#include <memory>
#include <vector>

namespace mlx_perf {

class CudaExecutionProbe {
 public:
  // `pass_started_at` is the steady_clock instant every other timestamp in
  // the pass is measured from. One slot per timed unit.
  CudaExecutionProbe(int slot_count, std::chrono::steady_clock::time_point pass_started_at);
  ~CudaExecutionProbe();
  CudaExecutionProbe(const CudaExecutionProbe&) = delete;
  CudaExecutionProbe& operator=(const CudaExecutionProbe&) = delete;

  // Call immediately before dispatching the unit at `position` on `stream`:
  // graph launches on that stream from this thread belong to it until the
  // stream is held for another unit or the mark below.
  void hold_buffer_for(int position, const mlx::core::Stream& stream);

  // Call immediately after dispatching the unit at `position`: its stream's
  // later launches no longer belong to it. Named after the Metal probe's end
  // marker, whose place it takes; CUDA needs no marker.
  void commit_end_marker_for(int position);

  // Call only after every unit is observed complete.
  std::vector<GpuExecutionWindow> read_windows() const;

  // Graph launches attributed to a unit and seen to run. Call only after
  // every unit is observed complete.
  int count_completed_command_buffers() const;

  struct ProbeState;  // defined in the .cpp

 private:
  std::unique_ptr<ProbeState> probe_state_;
  ProbeState* enclosing_probe_state_ = nullptr;
};

} // namespace mlx_perf
