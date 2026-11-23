#pragma once

// DOT writing: one concern only — emit a valid, styled Graphviz digraph that
// renders the captured plan the way Spark colors stages.

#include "graph_model.h"
#include "telemetry.h"

#include <string>

namespace mlx_plan {

// Renders the attributed plan: node labels carry op name, shape, dtype and
// duration; fill color encodes the stream; edge labels carry the bytes moved.
std::string write_plan_dot(
    const CapturedGraph& captured_graph,
    const RunTelemetry& run_telemetry,
    const std::string& report_title);

} // namespace mlx_plan
