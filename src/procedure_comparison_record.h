#pragma once

// Procedure comparison record: one concern only — express a procedure
// comparison as a JSON tree (`"kind": "procedure_comparison"`, schema version
// 1, described in docs/measurement-file.md): every round of both procedures,
// the verdicts, and each procedure's kernel launches grouped by Metal
// function. Pure: no files and no GPU.

#include "json_value.h"
#include "procedure_measurement.h"

namespace mlx_perf {

JsonValue build_procedure_comparison_record(const PairedProcedureMeasurement& paired_measurement,
                                            const ProcedureComparisons& procedure_comparisons);

} // namespace mlx_perf
