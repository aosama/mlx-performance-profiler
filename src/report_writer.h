#pragma once

// Report writing: one concern only — assemble the self-contained Spark-style
// HTML report from its sections and persist it (plus the DOT artifact, for
// anyone who wants the plan in Graphviz). Each section renders itself; this
// module only orders and saves them.

#include "report_inputs.h"

#include <string>

namespace mlx_perf {

struct ReportPaths {
  std::string html_path;
  std::string dot_path;
  // False when the HTML or DOT file could not be written (missing directory,
  // permissions, full disk); callers must not claim a report exists then.
  bool is_written = false;
};

// `measurement_document_json` is the same variant's measurement document
// (measurement_document.h, as text): the interactive graph view draws from
// it, so the report's graph and the measurement file show the same numbers.
ReportPaths write_report(
    const ReportInputs& report_inputs,
    const std::string& measurement_document_json,
    const std::string& output_directory,
    const std::string& plan_file_base);

// The task table matching the timeline's granularity: one row per node, or
// one per stage task. Shared by the static report and the live view.
std::string render_tasks_section(const ReportInputs& report_inputs);

// The full HTML document, without touching the filesystem (testable).
bool write_text_file(const std::string& file_path, const std::string& file_text);

std::string build_report_document(
    const ReportInputs& report_inputs,
    const std::string& measurement_document_json);

} // namespace mlx_perf
