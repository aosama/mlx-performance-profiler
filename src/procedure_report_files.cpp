#include "procedure_report_files.h"

#include "console_output.h"
#include "measurement_file.h"
#include "procedure_comparison_record.h"
#include "report_procedure_page.h"
#include "report_writer.h"

namespace mlx_perf {

bool write_procedure_report_files(const ProcedureReportRequest& report_request,
                                  const PairedProcedureMeasurement& paired_measurement,
                                  const ProcedureComparisons& procedure_comparisons) {
  const std::string page_path = report_request.output_directory + "/" + report_request.file_base + ".html";
  const bool is_page_written = write_text_file(
      page_path, build_procedure_comparison_document(report_request.page_title, paired_measurement,
                                                     procedure_comparisons));
  if (is_page_written) {
    print_progress_line("report: %s (%s vs %s)\n", page_path.c_str(),
                        paired_measurement.baseline.procedure_name.c_str(),
                        paired_measurement.candidate.procedure_name.c_str());
  } else {
    print_error_line("error: could not write %s\n", page_path.c_str());
  }
  const std::string record_path = write_json_file(
      build_procedure_comparison_record(paired_measurement, procedure_comparisons),
      report_request.output_directory, report_request.file_base);
  if (record_path.empty()) {
    print_error_line("error: could not write the procedure comparison %s.json\n", report_request.file_base.c_str());
  } else {
    print_progress_line("measurements: %s (procedure comparison)\n", record_path.c_str());
  }
  return is_page_written && !record_path.empty();
}

} // namespace mlx_perf
