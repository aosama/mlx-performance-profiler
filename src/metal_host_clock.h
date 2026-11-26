#pragma once

// Metal host clock: one concern only — the host time Metal stamps command
// buffers with (GPUStartTime / GPUEndTime, in seconds), read now, so GPU
// windows and host instants can be compared on one clock.

namespace mlx_perf {

// Seconds on the same clock as MTLCommandBuffer's GPUStartTime and
// GPUEndTime (mach_absolute_time, the clock CACurrentMediaTime reads),
// computed directly to avoid a QuartzCore call.
double current_metal_host_seconds();

} // namespace mlx_perf
