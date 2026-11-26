#pragma once

// Report components: one concern only — small HTML fragments reused by more
// than one report section (stacked latency bar, legend, class badge, the
// numeric and slice cells of the task tables).

#include "latency_breakdown.h"

#include <string>

namespace mlx_perf {

// Spark's task-assignment duration bar: one segment per slice, in causal
// order, widths proportional to the node's wall. Hover shows each slice.
std::string render_stacked_bar(const LatencyBreakdown& breakdown);
std::string render_slice_legend();
std::string render_class_badge(NodeClass node_class);

// A right-aligned table cell that sorts by `sort_value` and shows `shown`.
std::string render_number_cell(double sort_value, const std::string& shown);
std::string render_duration_cell(double microseconds);
// The six slice columns, header and body, in causal order. A "~" marks
// compute that is a host residual (no GPU window).
std::string render_slice_header_cells();
std::string render_slice_cells(const LatencyBreakdown& breakdown);

} // namespace mlx_perf
