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
// Honest limits:
//  - Granularity is the command buffer. In the timeline pass that is one node
//    (plus MLX's tiny event signal/wait commands). If a single primitive
//    overflows MLX's per-buffer op budget and MLX commits mid-node, only the
//    first buffer is seen and the window is an underestimate.
//  - It is the window the GPU spent on the buffer, including any cross-stream
//    event wait encoded at its head; the dependency share is separated later
//    from producer end times, not by the probe.
//  - CPU-device streams have no command buffer; they report no window.

#include "mlx/stream.h"

#include <chrono>
#include <vector>

namespace mlx_plan {

struct GpuExecutionWindow {
  bool is_measured = false;
  double started_microseconds = 0.0;   // since the pass anchor
  double finished_microseconds = 0.0;  // since the pass anchor
};

class MetalCommandBufferProbe {
 public:
  // `pass_started_at` is the steady_clock instant every other timestamp in
  // the pass is measured from; the probe anchors the Metal host clock to it.
  MetalCommandBufferProbe(
      int node_count, std::chrono::steady_clock::time_point pass_started_at);
  ~MetalCommandBufferProbe();
  MetalCommandBufferProbe(const MetalCommandBufferProbe&) = delete;
  MetalCommandBufferProbe& operator=(const MetalCommandBufferProbe&) = delete;

  // Call immediately before dispatching the node at `position` on `stream`.
  void hold_buffer_for(int position, const mlx::core::Stream& stream);

  // Call only after every node is observed complete.
  std::vector<GpuExecutionWindow> read_windows() const;

 private:
  // Opaque MTL::CommandBuffer* so this header stays free of metal-cpp.
  std::vector<void*> position_to_command_buffer_;
  double pass_anchor_host_seconds_ = 0.0;
};

} // namespace mlx_plan
