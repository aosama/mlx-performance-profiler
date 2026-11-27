#pragma once

// GPU execution window: one concern only — the span the GPU spent on one
// timed unit (a node, or a stage task), on the pass clock. Plain data that
// every backend's execution probe fills and the timelines read.

namespace mlx_perf {

struct GpuExecutionWindow {
  bool is_measured = false;
  // True when `finished_microseconds` came from an end marker (an upper
  // bound on the last submission's end) rather than the unit's own end.
  bool is_end_bounded_by_marker = false;
  double started_microseconds = 0.0;   // since the pass anchor
  double finished_microseconds = 0.0;  // since the pass anchor
};

} // namespace mlx_perf
