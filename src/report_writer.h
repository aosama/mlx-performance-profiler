#pragma once

// Report writing: one concern only — assemble the self-contained Spark-style
// HTML report from its sections and persist it (plus the DOT artifact).
// Each section renders itself; this module only orders and saves them.

#include "report_inputs.h"

#include <string>

namespace mlx_plan {

struct ReportPaths {
  std::string html_path;
  std::string dot_path;
  std::string svg_path; // empty when Graphviz was unavailable
  bool graphviz_available = false;
  // False when the HTML or DOT file could not be written (missing directory,
  // permissions, full disk); callers must not claim a report exists then.
  bool is_written = false;
  // The rendered DAG markup (empty without Graphviz), kept so a comparison
  // page can embed it without re-running the layout.
  std::string plan_svg;
};

ReportPaths write_report(
    const ReportInputs& report_inputs,
    const std::string& output_directory,
    const std::string& plan_file_base);

// The full HTML document, without touching the filesystem (testable).
bool write_text_file(const std::string& file_path, const std::string& file_text);

std::string build_report_document(
    const ReportInputs& report_inputs,
    const std::string& plan_svg,
    const std::string& plan_dot);

} // namespace mlx_plan
