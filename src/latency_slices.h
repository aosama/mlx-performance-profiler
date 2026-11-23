#pragma once

// Latency slices: one concern only — the single definition of the six
// StagePage-style slices (causal order, display label, CSS class, meaning),
// so the stacked bars, the Gantt and the legend can never drift apart.

#include "latency_breakdown.h"

#include <array>

namespace mlx_plan {

enum class LatencySlice {
  wait_on_cpu,
  wait_on_gpu,
  wait_on_dependency,
  scheduler_delay,
  compute,
  completion_latency,
};

// The order the slices occur in inside a node's window, which is also the
// stacking order of every bar.
inline constexpr std::array<LatencySlice, 6> CAUSAL_SLICE_ORDER = {
    LatencySlice::wait_on_cpu,
    LatencySlice::wait_on_gpu,
    LatencySlice::wait_on_dependency,
    LatencySlice::scheduler_delay,
    LatencySlice::compute,
    LatencySlice::completion_latency,
};

struct SliceStyle {
  const char* css_class;     // "slice-compute"; colors live in report_assets
  const char* short_label;   // table header / legend
  const char* spark_analog;  // the StagePage slice it mirrors
  const char* meaning;       // one-line tooltip text
};

const SliceStyle& slice_style(LatencySlice latency_slice);
// The slice a classification is named after, so a class badge wears the
// same color as the bar segment that earned it.
LatencySlice slice_for_class(NodeClass node_class);
double slice_microseconds(
    const LatencyBreakdown& breakdown, LatencySlice latency_slice);

} // namespace mlx_plan
