#pragma once

// Report summary section: one concern only — the Spark "job" view: headline
// cards, and where the nodes' time went in aggregate.

#include "report_inputs.h"

#include <string>

namespace mlx_plan {

std::string render_summary_section(const ReportInputs& report_inputs);

} // namespace mlx_plan
