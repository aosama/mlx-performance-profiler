#pragma once

// GPU backend vocabulary: one concern only — the words reports and console
// output use for what the GPU backend this library was built for measures
// and launches, so every other module stays backend-neutral and a macOS
// report reads exactly as it did before CUDA support.

namespace mlx_perf::gpu_vocabulary {

#if defined(MLX_PERF_GPU_BACKEND_CUDA)
// The measurement file's top-level "backend".
inline constexpr const char* BACKEND_NAME = "cuda";
inline constexpr const char* KERNEL_FUNCTION_HEADING = "CUDA kernel";
inline constexpr const char* GPU_TIMESTAMP_SOURCE = "CUPTI kernel timestamps";
inline constexpr const char* CONSOLE_TIMESTAMP_SOURCE = "CUPTI kernel start/end";
inline constexpr const char* CONSOLE_STAGE_NOTE = "; per stage, over every graph launch";
inline constexpr const char* KERNEL_COMPILATION = "CUDA kernel JIT compilation";
inline constexpr const char* TIMELINE_WINDOW_SENTENCE =
    " reads, through CUPTI, the start and end of every kernel each node's CUDA graph"
    " launch ran: compute is GPU-measured from the node's first kernel start to its"
    " last kernel end, and producers count as ready at their GPU end. Its per-node"
    " graph launches add submission cost a plain eval batches away";
inline constexpr const char* STAGE_WINDOW_SENTENCE =
    " A stage's GPU window runs from the first kernel start to the last kernel end"
    " over every CUDA graph launch the stage made (MLX launches a stage in several"
    " graphs when one would exceed its size budget; CUPTI reports them all).";
inline constexpr const char* WORK_GROUP_PLACEMENT =
    "A thread block runs on one streaming multiprocessor (SM), so a launch with fewer blocks than the GPU's ";
inline constexpr const char* COMPUTE_UNITS = "SMs";
inline constexpr const char* WORK_GROUPS_HEADING = "Blocks";
inline constexpr const char* COMPUTE_UNITS_HEADING = "SMs with work";
inline constexpr const char* COMPUTE_UNITS_TITLE = "Multiprocessors the launch with the fewest blocks can give work to";
#else
inline constexpr const char* BACKEND_NAME = "metal";
inline constexpr const char* KERNEL_FUNCTION_HEADING = "Metal function";
inline constexpr const char* GPU_TIMESTAMP_SOURCE = "Metal command-buffer timestamps";
inline constexpr const char* CONSOLE_TIMESTAMP_SOURCE = "Metal GPUStart/EndTime";
inline constexpr const char* CONSOLE_STAGE_NOTE = "; per stage, end bounded by a marker buffer";
inline constexpr const char* KERNEL_COMPILATION = "Metal kernel compilation";
inline constexpr const char* TIMELINE_WINDOW_SENTENCE =
    " reads each node's Metal command-buffer <code>GPUStartTime/GPUEndTime</code>:"
    " compute is GPU-measured at command-buffer granularity, and producers count as"
    " ready at their GPU end. Its per-node command buffers add submission cost a plain"
    " eval batches away";
inline constexpr const char* STAGE_WINDOW_SENTENCE =
    " A stage's GPU window runs from its first command buffer's GPU start to the end of"
    " its last buffer, bounded by an empty marker buffer queued behind it (MLX commits"
    " mid-stage when a buffer exceeds its op budget, and those middle buffers are not"
    " observable) and by host-observed completion.";
inline constexpr const char* WORK_GROUP_PLACEMENT =
    "A threadgroup runs on one GPU core, so a launch with fewer threadgroups than the GPU's ";
inline constexpr const char* COMPUTE_UNITS = "cores";
inline constexpr const char* WORK_GROUPS_HEADING = "Threadgroups";
inline constexpr const char* COMPUTE_UNITS_HEADING = "Cores with work";
inline constexpr const char* COMPUTE_UNITS_TITLE = "GPU cores the launch with the fewest threadgroups can give work to";
#endif

} // namespace mlx_perf::gpu_vocabulary
