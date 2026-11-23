#pragma once

// Report Gantt section: one concern only — the per-stream timeline (Spark's
// event timeline / task-assignment Gantt) as inline SVG, plus the
// per-stream summary table that reads alongside it.

#include "report_inputs.h"

#include <string>

namespace mlx_plan {

std::string render_gantt_section(const ReportInputs& report_inputs);

} // namespace mlx_plan
