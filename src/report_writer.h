#pragma once

// Report writing: one concern only — assemble and persist the self-contained
// HTML report (summary cards, the stage-colored DAG, the per-node stage
// table) from captured plan + telemetry.

#include "graph_model.h"
#include "telemetry.h"

#include <string>

namespace mlx_plan {

struct ReportPaths {
  std::string html_path;
  std::string dot_path;
  std::string svg_path; // empty when Graphviz was unavailable
  bool graphviz_available = false;
};

ReportPaths write_report(
    const CapturedGraph& captured_graph,
    const RunTelemetry& run_telemetry,
    const std::string& output_directory,
    const std::string& plan_file_base,
    const std::string& report_title);

} // namespace mlx_plan
