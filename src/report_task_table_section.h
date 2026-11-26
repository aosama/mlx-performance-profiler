#pragma once

// Report task table section: one concern only — Spark's per-task table: one
// sortable row per executed node with its slices and stacked latency bar.

#include "report_inputs.h"

#include <string>

namespace mlx_perf {

std::string render_task_table_section(const ReportInputs& report_inputs);

} // namespace mlx_perf
