#pragma once

// Dispatch occupancy: one concern only — describe a kernel launch (its grid
// and threadgroup shape, and its kernel's limits) and say how much of the GPU
// it could keep busy. Pure arithmetic, no GPU API: the recorders that observe
// launches live in metal_dispatch_recorder and cuda_dispatch_recorder.
//
// The words are Metal's; on CUDA a threadgroup is a thread block, a GPU core
// is a streaming multiprocessor, and a SIMD group is a warp.
//
// Why core coverage, and nothing finer: a threadgroup runs on one GPU core,
// so a launch with fewer threadgroups than the GPU has cores leaves cores with
// nothing to do, whatever else is true. That bound is exact on every Mac, and
// on NVIDIA GPUs with blocks and multiprocessors. How
// many threadgroups one core can hold at once depends on per-core thread and
// memory limits Apple does not document, so this module does not guess them:
// a launch that covers every core can still be limited in other ways.

#include <cstdint>
#include <string>

namespace mlx_perf {

struct GridSize {
  std::uint64_t width = 1;
  std::uint64_t height = 1;
  std::uint64_t depth = 1;
};

// What a launch's grid counts: dispatchThreadgroups passes threadgroups,
// dispatchThreads passes threads (Metal forms the threadgroups, rounding up).
// CUDA launches always count blocks, so they record threadgroups.
enum class DispatchGridUnit { threadgroups, threads };

struct KernelDispatch {
  // The kernel function: on Metal the function the pipeline was compiled
  // from, on CUDA the kernel's mangled name as the driver reports it, e.g.
  // "gemv_t_bfloat16_bm8_bn32_sm1_sn32_tm4_tn4" or "custom_kernel_rmsnorm".
  // Empty on Metal when MLX compiled the pipeline before the recorder was
  // installed.
  std::string kernel_function_name;
  DispatchGridUnit grid_unit = DispatchGridUnit::threadgroups;
  GridSize grid;                  // as passed to the GPU, in grid_unit
  GridSize threads_per_threadgroup;
  // The pipeline's limits; 0 when the pipeline was not seen being set.
  std::uint64_t max_threads_per_threadgroup = 0;
  std::uint64_t thread_execution_width = 0;  // the SIMD group width
  // Static plus dynamically set threadgroup memory, in bytes.
  std::uint64_t threadgroup_memory_bytes = 0;
};

std::uint64_t grid_volume(const GridSize& grid);

// The threadgroup grid the launch forms: the grid itself for
// dispatchThreadgroups; for dispatchThreads, the thread grid divided by the
// threadgroup size, rounded up per dimension. A zero-sized threadgroup forms
// no threadgroups (Metal rejects such a launch).
GridSize threadgroup_grid(const KernelDispatch& kernel_dispatch);

std::uint64_t count_threadgroups(const KernelDispatch& kernel_dispatch);

// The share of GPU cores the launch can give work to: threadgroups over
// cores, at most 1. Negative when the core count is unknown (0 or less).
double gpu_core_coverage(std::uint64_t threadgroup_count, int gpu_core_count);

// True when the launch has at least one threadgroup per GPU core. False when
// it cannot, or when the core count is unknown (nothing is claimed then).
bool can_fill_gpu(std::uint64_t threadgroup_count, int gpu_core_count);

const char* dispatch_grid_unit_name(DispatchGridUnit grid_unit);

} // namespace mlx_perf
