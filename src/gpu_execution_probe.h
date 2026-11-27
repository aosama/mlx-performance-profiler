#pragma once

// GPU execution probe: one concern only — name the probe the timeline passes
// use to read each timed unit's GPU execution window, for the GPU backend
// this library was configured for.
//
// Why a compile-time choice and not a virtual interface: a build links one
// MLX, and an MLX build has exactly one GPU backend, so there is never a
// second probe to choose at run time. Every probe has the same surface:
//
//   Probe(int slot_count, std::chrono::steady_clock::time_point pass_started_at);
//   void hold_buffer_for(int position, const mlx::core::Stream& stream);   // before dispatch
//   void commit_end_marker_for(int position);                              // after dispatch
//   std::vector<GpuExecutionWindow> read_windows() const;                   // once complete
//   int count_completed_command_buffers() const;                            // once complete
//
// Metal: metal_command_buffer_probe (command-buffer GPUStartTime/GPUEndTime).
// CUDA: cuda_execution_probe (CUPTI activity records of each unit's graph
// launches).

#include "gpu_execution_window.h"

#if defined(MLX_PERF_GPU_BACKEND_CUDA)
#include "cuda_execution_probe.h"
namespace mlx_perf {
using GpuExecutionProbe = CudaExecutionProbe;
} // namespace mlx_perf
#else
#include "metal_command_buffer_probe.h"
namespace mlx_perf {
using GpuExecutionProbe = MetalCommandBufferProbe;
} // namespace mlx_perf
#endif
