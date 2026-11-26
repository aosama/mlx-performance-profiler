#pragma once

// Procedure report files: one concern only — write a procedure comparison as
// `<output_directory>/<file_base>.html` (the page) and `.json` (the record),
// reporting each file written or failed through console_output.

#include "procedure_measurement.h"

#include <string>

namespace mlx_perf {

struct ProcedureReportRequest {
  std::string page_title;
  std::string output_directory;
  std::string file_base;
};

// False when either file could not be written.
bool write_procedure_report_files(const ProcedureReportRequest& report_request,
                                  const PairedProcedureMeasurement& paired_measurement,
                                  const ProcedureComparisons& procedure_comparisons);

} // namespace mlx_perf
