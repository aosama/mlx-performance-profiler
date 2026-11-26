#pragma once

// Metal command-buffer probe: one concern only — read the GPU's own
// execution window for each dispatched node, on the host clock.
//
// Why this works without touching MLX (0.32.4, backend/metal/eval.cpp):
// every async_eval ends in gpu::finalize, which commits the stream's current
// MTLCommandBuffer and starts a fresh one. So the buffer a stream holds just
// BEFORE a node is dispatched is exactly the buffer that node is encoded
// into. The probe retains it, and once the node completes reads
// GPUStartTime / GPUEndTime — host-timebase seconds (mach_absolute_time),
// which convert onto the pass clock with a single anchor sample.
//
// Multi-buffer units (per-stage pass): MLX commits mid-eval whenever a
// buffer exceeds its op/MB budget (device.cpp, get_max_ops_mb_per_buffer),
// so one async_eval can span several buffers the probe never sees. For those
// units the caller commits an empty END MARKER buffer on the same queue
// right after the dispatch. A queue runs its buffers in commit order, so the
// marker's GPUStartTime is when the GPU was done with the unit's LAST
// buffer (an upper bound: it adds the marker's own scheduling gap, measured
// at ~25 µs, which the caller tightens with host-observed completion).
//
// Honest limits:
//  - Granularity is the command buffer. Without a marker, if a primitive
//    overflows MLX's per-buffer op budget and MLX commits mid-node, only the
//    first buffer is seen and the window is an underestimate.
//  - It is the window the GPU spent on the buffer, including any cross-stream
//    event wait encoded at its head; the dependency share is separated later
//    from producer end times, not by the probe.
//  - CPU-device streams have no command buffer; they report no window.

#include "mlx/stream.h"

#include <chrono>
#include <vector>

namespace mlx_perf {

struct GpuExecutionWindow {
  bool is_measured = false;
  // True when `finished_microseconds` came from an end marker (an upper
  // bound on the last buffer's end) rather than the held buffer's own end.
  bool is_end_bounded_by_marker = false;
  double started_microseconds = 0.0;   // since the pass anchor
  double finished_microseconds = 0.0;  // since the pass anchor
};

class MetalCommandBufferProbe {
 public:
  // `pass_started_at` is the steady_clock instant every other timestamp in
  // the pass is measured from; the probe anchors the Metal host clock to it.
  // One slot per timed unit (a node, or a stage task).
  MetalCommandBufferProbe(
      int slot_count, std::chrono::steady_clock::time_point pass_started_at);
  ~MetalCommandBufferProbe();
  MetalCommandBufferProbe(const MetalCommandBufferProbe&) = delete;
  MetalCommandBufferProbe& operator=(const MetalCommandBufferProbe&) = delete;

  // Call immediately before dispatching the unit at `position` on `stream`.
  void hold_buffer_for(int position, const mlx::core::Stream& stream);

  // Call immediately AFTER dispatching the unit at `position`: queues an
  // empty buffer behind it whose GPU start bounds the unit's GPU end. A
  // no-op when no GPU buffer is held for `position` (CPU streams).
  void commit_end_marker_for(int position);

  // Call only after every node is observed complete.
  std::vector<GpuExecutionWindow> read_windows() const;

  // Distinct MLX command buffers held and seen to complete (end markers are
  // this probe's own and are not counted). Exact when every unit fits one
  // buffer; a lower bound when MLX commits mid-unit, since those middle
  // buffers are never held. Call only after every unit is observed complete.
  int count_completed_command_buffers() const;

 private:
  // Opaque MTL::CommandBuffer* so this header stays free of metal-cpp.
  std::vector<void*> position_to_command_buffer_;
  std::vector<void*> position_to_end_marker_buffer_;
  double pass_anchor_host_seconds_ = 0.0;
};

} // namespace mlx_perf
