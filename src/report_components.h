#pragma once

// Report components: one concern only — small HTML fragments reused by more
// than one report section (stacked latency bar, legend, class badge).

#include "latency_breakdown.h"

#include <string>

namespace mlx_plan {

// Spark's task-assignment duration bar: one segment per slice, in causal
// order, widths proportional to the node's wall. Hover shows each slice.
std::string render_stacked_bar(const LatencyBreakdown& breakdown);
std::string render_slice_legend();
std::string render_class_badge(NodeClass node_class);

} // namespace mlx_plan
