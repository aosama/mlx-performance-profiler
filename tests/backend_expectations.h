#pragma once

// Backend expectations: one concern only — what the shared tests may expect
// from the GPU backend and operating system this build links, where the two
// backends differ by design rather than by defect.

namespace backend_expectations {

#if defined(MLX_PERF_GPU_BACKEND_CUDA)
// Linux has no system-wide wired-memory line, so the readings are
// unavailable (zeros in the structs, null in the JSON).
inline constexpr bool HAS_SYSTEM_WIRED_MEMORY = false;
// A node that launches no kernel (a broadcast that only describes a view)
// makes no CUDA graph launch, so it has no GPU window.
inline constexpr bool DOES_EVERY_GPU_NODE_SUBMIT_WORK = false;
#else
// Every Metal async_eval commits a command buffer, which carries GPU
// timestamps even when it encodes nothing.
inline constexpr bool HAS_SYSTEM_WIRED_MEMORY = true;
inline constexpr bool DOES_EVERY_GPU_NODE_SUBMIT_WORK = true;
#endif

} // namespace backend_expectations
