#pragma once

// GPU host clock: one concern only — the host time the GPU's own timestamps
// are reported on, read now, so GPU windows and host instants can be
// compared on one clock.
//
// One implementation per GPU backend, chosen at configure time:
//  - Metal (metal_host_clock.cpp): mach_absolute_time, the clock
//    MTLCommandBuffer's GPUStartTime and GPUEndTime are stamped on.
//  - CUDA (cuda_host_clock.cpp): CLOCK_MONOTONIC, the clock this library
//    hands CUPTI as its timestamp source (cupti_session), so every kernel
//    record CUPTI delivers is already on it.

namespace mlx_perf {

// Seconds on the clock the backend's GPU timestamps are reported on.
double current_gpu_host_seconds();

} // namespace mlx_perf
