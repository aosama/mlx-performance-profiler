#include "report_summary_section.h"

#include "latency_slices.h"
#include "report_components.h"
#include "text_format.h"

#include <sstream>

namespace mlx_plan {
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
      "Job duration", "One warm eval, streams overlapping, uninstrumented.");
  append_card(cards_html,
      format_microseconds(gpu_timeline.measured_gpu_compute_microseconds),
      "GPU execution (measured)",
      std::to_string(gpu_timeline.gpu_measured_node_count) + "/" +
          std::to_string(gpu_timeline.nodes.size()) +
          " nodes measured from Metal command-buffer timestamps.");
  append_card(cards_html,
      format_microseconds(gpu_timeline.cross_stream_stall_microseconds),
      "Cross-stream stall",
      "Summed 'stalled' slices: Spark fetchWaitTime. Stream-time, can exceed wall.");
  append_card(cards_html,
      format_microseconds(gpu_timeline.timeline_wall_microseconds),
      "Instrumented duration",
      "Timeline pass (one command buffer per node); " +
          format_microseconds(gpu_timeline.timeline_wall_microseconds -
                              job_totals.job_wall_microseconds) +
          " over the job is instrumentation cost.");
  append_card(cards_html,
      format_microseconds(job_totals.sum_of_stages_microseconds),
      "Sum of stages", "Serial pass: each node synchronized alone, no overlap.");
  append_card(cards_html,
      format_bytes(report_inputs.serial_telemetry.run_peak_memory_bytes),
      "Peak memory", "Highest per-node peak during the serial pass.");
  append_card(cards_html,
      std::to_string(gpu_timeline.nodes.size()) + " · " +
          std::to_string(gpu_timeline.streams.size()),
      "Nodes · streams", "Primitives executed, and the streams they ran on.");
  cards_html << "</div>";
  return cards_html.str();
}

// Sums every node's slices into one breakdown: the job-level answer to
// "what were the nodes doing", with the same bar the task table uses.
LatencyBreakdown aggregate_breakdown(const GpuTimeline& gpu_timeline) {
  LatencyBreakdown aggregate;
  for (const NodeTimeline& node_timeline : gpu_timeline.nodes) {
    const LatencyBreakdown& breakdown = node_timeline.breakdown;
    aggregate.wait_on_cpu_microseconds += breakdown.wait_on_cpu_microseconds;
    aggregate.wait_on_gpu_microseconds += breakdown.wait_on_gpu_microseconds;
    aggregate.wait_on_dependency_microseconds +=
        breakdown.wait_on_dependency_microseconds;
    aggregate.scheduler_delay_microseconds +=
        breakdown.scheduler_delay_microseconds;
    aggregate.compute_microseconds += breakdown.compute_microseconds;
    aggregate.completion_latency_microseconds +=
        breakdown.completion_latency_microseconds;
    aggregate.wall_microseconds += breakdown.wall_microseconds;
  }
  return aggregate;
}

std::string render_time_distribution(const GpuTimeline& gpu_timeline) {
  const LatencyBreakdown aggregate = aggregate_breakdown(gpu_timeline);
  std::ostringstream distribution_html;
  distribution_html << "<h2>Where node time went</h2>" << render_slice_legend()
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
  distribution_html << "<br>Summed over all nodes of the timeline pass. Node"
                    << " windows overlap across streams, so the total is"
                    << " node-time, not wall-clock.</p>";
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

} // namespace mlx_plan
