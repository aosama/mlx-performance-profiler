#include "report_summary_section.h"

#include "gpu_backend_vocabulary.h"
#include "latency_slices.h"
#include "report_components.h"
#include "text_format.h"
#include "timed_units.h"

#include <sstream>

namespace mlx_perf {
namespace {

void append_card(
    std::ostringstream& cards_html,
    const std::string& card_value,
    const std::string& card_label,
    const std::string& card_detail) {
  cards_html << "<div class=\"card\"><div class=\"card-value\">"
             << escape_html_text(card_value) << "</div><div class=\"card-label\">"
             << escape_html_text(card_label) << "</div><div class=\"card-detail\">"
             << escape_html_text(card_detail) << "</div></div>";
}

std::string render_cards(const ReportInputs& report_inputs) {
  const GpuTimeline& gpu_timeline = report_inputs.gpu_timeline;
  const JobTotals& job_totals = report_inputs.job_totals;
  std::ostringstream cards_html;
  cards_html << "<div class=\"cards\">";
  append_card(cards_html, format_microseconds(job_totals.job_wall_microseconds),
      "Job duration",
      "Median of " + std::to_string(job_totals.job_wall_sample_count) +
          " warm, uninstrumented evals (range " +
          format_microseconds(job_totals.job_wall_min_microseconds) + " – " +
          format_microseconds(job_totals.job_wall_max_microseconds) + ").");
  append_card(cards_html,
      format_microseconds(gpu_timeline.measured_gpu_compute_microseconds),
      "GPU execution (measured)",
      std::to_string(gpu_timeline.gpu_measured_node_count) + "/" +
          std::to_string(gpu_timeline.nodes.size()) +
          " nodes measured from " + gpu_vocabulary::GPU_TIMESTAMP_SOURCE +
          (gpu_timeline.granularity == TimelineGranularity::per_stage
               ? std::string(", per stage (includes GPU gaps while the host"
                             " encoded the stage).")
               : std::string(".")));
  append_card(cards_html,
      format_microseconds(gpu_timeline.cross_stream_stall_microseconds),
      "Cross-stream stall",
      "Summed 'stalled' slices: Spark fetchWaitTime. Stream-time, can exceed wall.");
  const double timeline_wall = gpu_timeline.timeline_wall_microseconds;
  const std::string pass_description = gpu_timeline.granularity ==
          TimelineGranularity::per_stage
      ? "Timeline pass (one async_eval per stage, " +
            std::to_string(gpu_timeline.stages.size()) + " stage tasks)"
      : std::string("Timeline pass (one command buffer per node)");
  append_card(cards_html, format_microseconds(timeline_wall),
      "Instrumented duration",
      is_within_job_wall_range(job_totals, timeline_wall)
          ? pass_description + "; inside the job-duration range: no measurable"
                               " instrumentation cost."
          : pass_description + "; " +
                format_microseconds(timeline_wall - job_totals.job_wall_microseconds) +
                " vs the job is instrumentation cost.");
  append_card(cards_html,
      format_microseconds(job_totals.sum_of_stages_microseconds),
      "Sum of stages", "Serial pass: each node synchronized alone, no overlap.");
  append_card(cards_html,
      format_bytes(report_inputs.serial_telemetry.run_peak_memory_bytes),
      "Peak memory",
      "Highest per-node peak during the serial pass, above what was already allocated when it"
      " began.");
  append_card(cards_html,
      std::to_string(gpu_timeline.nodes.size()) + " · " +
          std::to_string(gpu_timeline.streams.size()),
      "Nodes · streams", "Primitives executed, and the streams they ran on.");
  cards_html << "</div>";
  return cards_html.str();
}

std::string render_time_distribution(const GpuTimeline& gpu_timeline) {
  const LatencyBreakdown aggregate = aggregate_breakdown(collect_timed_units(gpu_timeline));
  const std::string unit_noun = timed_unit_noun(gpu_timeline);
  std::ostringstream distribution_html;
  distribution_html << "<h2>Where " << unit_noun << " time went</h2>"
                    << render_slice_legend()
                    << render_stacked_bar(aggregate) << "<p class=\"note\">";
  for (const LatencySlice latency_slice : CAUSAL_SLICE_ORDER) {
    const double slice_total = slice_microseconds(aggregate, latency_slice);
    const double share_percent = aggregate.wall_microseconds > 0.0
        ? 100.0 * slice_total / aggregate.wall_microseconds
        : 0.0;
    char share_text[24];
    std::snprintf(share_text, sizeof(share_text), "%.1f%%", share_percent);
    distribution_html << "<b>" << slice_style(latency_slice).short_label
                      << "</b> " << format_microseconds(slice_total) << " ("
                      << share_text << ") &nbsp; ";
  }
  distribution_html << "<br>Summed over every " << unit_noun << " of the timeline"
                    << " pass. Windows overlap across streams, so the total is "
                    << unit_noun << "-time, not wall-clock.</p>";
  return distribution_html.str();
}

} // namespace

std::string render_summary_section(const ReportInputs& report_inputs) {
  std::ostringstream section_html;
  section_html << "<section id=\"job\"><h2>Job summary</h2>"
               << render_cards(report_inputs)
               << render_time_distribution(report_inputs.gpu_timeline)
               << "</section>";
  return section_html.str();
}

} // namespace mlx_perf
