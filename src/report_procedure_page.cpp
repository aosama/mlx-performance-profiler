#include "report_procedure_page.h"

#include "kernel_launch_summary.h"
#include "report_assets.h"
#include "report_launch_comparison_section.h"
#include "report_verdict_section.h"
#include "text_format.h"

#include <sstream>

namespace mlx_perf {
namespace {

const char* verdict_css_class(ComparisonVerdict verdict) {
  if (verdict == ComparisonVerdict::candidate_faster) return "faster";
  if (verdict == ComparisonVerdict::candidate_slower) return "slower";
  return "within-spread";
}

std::string render_verdict(const PairedProcedureMeasurement& paired_measurement,
                           const ProcedureComparisons& procedure_comparisons) {
  const std::string& baseline_name = paired_measurement.baseline.procedure_name;
  const std::string& candidate_name = paired_measurement.candidate.procedure_name;
  std::ostringstream section_html;
  section_html << "<section id=\"verdict\"><h2>Verdict</h2><p class=\"verdict "
               << verdict_css_class(procedure_comparisons.duration.verdict) << "\">"
               << escape_html_text(describe_verdict(baseline_name, candidate_name, procedure_comparisons.duration))
               << "</p>"
               << render_verdict_table(baseline_name, candidate_name,
                                       {{"Procedure duration", &procedure_comparisons.duration},
                                        {"GPU busy", &procedure_comparisons.gpu_busy},
                                        {"GPU idle (waiting on the host)", &procedure_comparisons.gpu_idle}})
               << "<p class=\"note\">Each round runs both procedures, flipping which goes first; the verdict"
               << " is judged on the per-round differences, so drift both share within a round cancels, and"
               << " a difference counts only when the whole interval lies on one side of zero (at least 10 of"
               << " 12 rounds). A run is timed from run() starting until every command buffer it committed has"
               << " completed. GPU busy is the time at least one of those buffers was on the GPU; GPU idle is"
               << " the rest of the run, when the GPU had nothing of the procedure's to do. Every sample is in"
               << " the comparison's JSON file.</p></section>";
  return section_html.str();
}

double median_of(const std::vector<double>& round_values) {
  return summarize_samples(round_values).median;
}

std::string render_handoff_row(const ProcedureMeasurement& procedure_measurement) {
  const double median_duration = median_of(collect_round_values(procedure_measurement, &ProcedureRun::duration_microseconds));
  const double median_busy = median_of(collect_round_values(procedure_measurement, &ProcedureRun::gpu_busy_microseconds));
  std::ostringstream row_html;
  row_html << "<tr><td>" << escape_html_text(procedure_measurement.procedure_name) << "</td><td class=\"number\">"
           << median_of(collect_round_counts(procedure_measurement, &ProcedureRun::command_buffer_count))
           << "</td><td class=\"number\">"
           << median_of(collect_round_counts(procedure_measurement, &ProcedureRun::blocking_wait_count))
           << "</td><td class=\"number\">"
           << format_microseconds(median_of(
                  collect_round_values(procedure_measurement, &ProcedureRun::blocking_wait_microseconds)))
           << "</td><td class=\"number\">" << format_microseconds(median_busy) << " of "
           << format_microseconds(median_duration) << " (" << static_cast<int>(
                  median_duration > 0.0 ? 100.0 * median_busy / median_duration + 0.5 : 0.0)
           << "%)</td><td class=\"number\">" << procedure_measurement.observed_launches.size() << "</td></tr>";
  return row_html.str();
}

std::string render_handoffs(const PairedProcedureMeasurement& paired_measurement) {
  std::ostringstream section_html;
  section_html << "<section id=\"handoffs\"><h2>Hand-offs per run</h2><div class=\"table-wrap\"><table>"
               << "<thead><tr><th>Procedure</th><th>Command buffers</th><th>Blocking waits</th>"
               << "<th>Host time blocked</th><th>GPU busy of the run</th><th>Kernel launches</th></tr></thead><tbody>"
               << render_handoff_row(paired_measurement.baseline)
               << render_handoff_row(paired_measurement.candidate)
               << "</tbody></table></div><p class=\"note\">Medians over the rounds; kernel launches are from"
               << " one extra observed run of each. A blocking wait is an eval or synchronize that stopped the"
               << " host until the GPU finished; each one is a hand-off in which neither side can run ahead,"
               << " so their number is often the finding. Command buffers are MLX's submissions to the GPU.</p>"
               << "</section>";
  return section_html.str();
}

} // namespace

std::string build_procedure_comparison_document(const std::string& page_title,
                                                const PairedProcedureMeasurement& paired_measurement,
                                                const ProcedureComparisons& procedure_comparisons) {
  const int gpu_core_count = paired_measurement.gpu_device_facts.gpu_core_count;
  std::ostringstream page_html;
  page_html << "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>"
            << escape_html_text(page_title) << "</title><style>" << report_stylesheet()
            << "</style></head><body><header><h1>" << escape_html_text(page_title) << "</h1><p>"
            << "Two procedures, each run whole: " << escape_html_text(paired_measurement.baseline.procedure_name)
            << " vs " << escape_html_text(paired_measurement.candidate.procedure_name) << "</p></header>"
            << render_verdict(paired_measurement, procedure_comparisons) << render_handoffs(paired_measurement)
            << render_launch_comparison_section(
                   paired_measurement.baseline.procedure_name,
                   summarize_launches_by_function(paired_measurement.baseline.observed_launches, gpu_core_count),
                   paired_measurement.candidate.procedure_name,
                   summarize_launches_by_function(paired_measurement.candidate.observed_launches, gpu_core_count),
                   gpu_core_count, "Metal function")
            << "</body></html>\n";
  return page_html.str();
}

} // namespace mlx_perf
