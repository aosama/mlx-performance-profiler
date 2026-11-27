#pragma once

// GPU busy time: one concern only — how long the GPU was busy inside a time
// span, given the GPU windows of the command buffers that ran in it. Pure
// arithmetic, no Metal.
//
// Why a union and not a sum: command buffers on different queues (streams)
// run at the same time, and summing their windows would count the GPU busy
// twice over the same microseconds. The union counts each busy microsecond
// once, so span minus busy is the time the GPU had nothing to run: time it
// waited on the host.

#include <vector>

namespace mlx_perf {

struct GpuWindow {
  double started_seconds = 0.0;   // GPU start, on the GPU host clock (gpu_host_clock)
  double finished_seconds = 0.0;  // GPU end, on the same clock
};

// The microseconds inside [span_started_seconds, span_finished_seconds]
// covered by at least one window. Windows outside the span are clipped; an
// inverted or empty window or span counts as nothing.
double measure_gpu_busy_microseconds(std::vector<GpuWindow> gpu_windows,
                                     double span_started_seconds, double span_finished_seconds);

} // namespace mlx_perf
