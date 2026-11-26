#include "live_view_renderer.h"

#include "report_comparison_page.h"
#include "report_verdict_section.h"
#include "report_gantt_section.h"
#include "report_memory_section.h"
#include "report_summary_section.h"
#include "report_writer.h"
#include "text_format.h"

#include <sstream>

namespace mlx_perf {
namespace {

ReportInputs make_report_inputs(const std::string& job_title,
                                const VariantMeasurement& variant_measurement) {
  return ReportInputs{job_title + " (" + variant_measurement.variant_name + ")",
                      variant_measurement.timeline_graph,
                      variant_measurement.gpu_timeline,
                      variant_measurement.serial_telemetry,
                      variant_measurement.job_totals};
}

std::string variant_heading(const VariantMeasurement& variant_measurement) {
  return "<h2 class=\"variant-heading\">" +
      escape_html_text(variant_measurement.variant_name) + "</h2>";
}

} // namespace

RenderedViews render_live_views(
    const std::string& job_title,
    const std::vector<const VariantMeasurement*>& measured_variants) {
  std::ostringstream job_html;
  std::ostringstream stages_html;
  std::ostringstream tasks_html;
  std::ostringstream memory_html;
  if (measured_variants.size() >= 2) {
    job_html << "<section><h2>Comparison</h2>"
             << render_comparison_metrics(*measured_variants[0], *measured_variants[1])
             << "</section>";
  }
  for (const VariantMeasurement* variant_measurement : measured_variants) {
    const ReportInputs report_inputs = make_report_inputs(job_title, *variant_measurement);
    job_html << variant_heading(*variant_measurement)
             << render_summary_section(report_inputs);
    stages_html << variant_heading(*variant_measurement)
                << render_gantt_section(report_inputs);
    tasks_html << variant_heading(*variant_measurement)
               << render_tasks_section(report_inputs);
    memory_html << "<section>" << variant_heading(*variant_measurement)
                << render_memory_section(*variant_measurement) << "</section>";
  }
  return {job_html.str(), stages_html.str(), tasks_html.str(), memory_html.str()};
}

RenderedViews render_live_views(const std::string& job_title,
                                const PairedMeasurement& paired_measurement) {
  RenderedViews rendered_views = render_live_views(
      job_title, {&paired_measurement.baseline, &paired_measurement.candidate});
  rendered_views.job_html =
      "<section><h2>Verdict</h2>" +
      render_verdict_section(paired_measurement.baseline.variant_name,
                             paired_measurement.candidate.variant_name,
                             compare_pair(paired_measurement),
                             paired_measurement.output_agreement) +
      "</section>" + rendered_views.job_html;
  return rendered_views;
}

} // namespace mlx_perf
