#pragma once

// Work limits section: one concern only — the report's "Data and arithmetic"
// section: this machine's two ceilings, and for every timed unit (node, or
// stage task in a stage-timed run) how much of its GPU time moving its data
// and doing its arithmetic would each take at those ceilings, and which one
// held it back.

#include "report_inputs.h"

#include <string>

namespace mlx_perf {

std::string render_work_limits_section(const ReportInputs& report_inputs);

} // namespace mlx_perf
