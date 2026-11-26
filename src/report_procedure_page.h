#pragma once

// Procedure comparison page: one concern only — the HTML page comparing two
// procedure jobs: the verdict on their duration, GPU busy and GPU idle time;
// their hand-offs between host and GPU per run; and their kernel launches
// side by side, grouped by Metal function (a procedure with hundreds of
// evaluations has no one graph to draw).

#include "procedure_measurement.h"

#include <string>

namespace mlx_perf {

std::string build_procedure_comparison_document(const std::string& page_title,
                                                const PairedProcedureMeasurement& paired_measurement,
                                                const ProcedureComparisons& procedure_comparisons);

} // namespace mlx_perf
