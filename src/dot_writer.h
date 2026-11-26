#pragma once

// DOT writing: one concern only — emit a valid, styled Graphviz digraph that
// renders the captured plan the way Spark colors stages.

#include "gpu_timeline.h"
#include "graph_model.h"

#include <string>

namespace mlx_perf {

// Node labels carry op, shape/dtype, measured GPU compute and class; fill
// color encodes the stream; a red outline marks nodes classified as stalled
// on another stream; edge labels carry the bytes moved. A per-stage timeline
// boxes each stage as a dashed cluster and labels nodes with their stage.
std::string write_plan_dot(
    const CapturedGraph& captured_graph,
    const GpuTimeline& gpu_timeline,
    const std::string& report_title);

} // namespace mlx_perf
