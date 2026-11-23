#include "report_comparison_page.h"

#include "report_assets.h"
#include "report_gantt_section.h"
#include "report_components.h"
#include "text_format.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <sstream>

namespace mlx_plan {
namespace {

constexpr std::string_view COMPILED_PREFIX = "Compiled";

struct MetricRow {
  const char* metric_name;
  double baseline_value;
  double candidate_value;
  bool is_duration;  // false → a count
};

std::string format_metric(double metric_value, bool is_duration) {
  return is_duration ? format_microseconds(metric_value)
                     : std::to_string(static_cast<long long>(metric_value));
}

// Lower is better for every metric listed, so a negative change is an
// improvement; the sign is shown, never hidden behind a color alone.
std::string format_change(double baseline_value, double candidate_value) {
  if (baseline_value == 0.0) {
    return "—";
  }
  char change_text[32];
  std::snprintf(change_text, sizeof(change_text), "%+.1f%%",
                100.0 * (candidate_value - baseline_value) / baseline_value);
  return change_text;
}

std::size_t count_timed_units(const GpuTimeline& gpu_timeline) {
  return gpu_timeline.granularity == TimelineGranularity::per_stage
      ? gpu_timeline.stages.size()
      : gpu_timeline.nodes.size();
}

std::vector<MetricRow> collect_metric_rows(
    const VariantMeasurement& baseline, const VariantMeasurement& candidate) {
  return {
      {"Job duration (median)", baseline.job_totals.job_wall_microseconds,
       candidate.job_totals.job_wall_microseconds, true},
      {"GPU execution (measured)",
       baseline.gpu_timeline.measured_gpu_compute_microseconds,
       candidate.gpu_timeline.measured_gpu_compute_microseconds, true},
      {"Cross-stream stall", baseline.gpu_timeline.cross_stream_stall_microseconds,
       candidate.gpu_timeline.cross_stream_stall_microseconds, true},
      {"Instrumented duration", baseline.gpu_timeline.timeline_wall_microseconds,
       candidate.gpu_timeline.timeline_wall_microseconds, true},
      {"Sum of stages", baseline.job_totals.sum_of_stages_microseconds,
       candidate.job_totals.sum_of_stages_microseconds, true},
      {"Executed nodes", static_cast<double>(baseline.gpu_timeline.nodes.size()),
       static_cast<double>(candidate.gpu_timeline.nodes.size()), false},
      {"Timed units (nodes, or stage tasks)",
       static_cast<double>(count_timed_units(baseline.gpu_timeline)),
       static_cast<double>(count_timed_units(candidate.gpu_timeline)), false},
  };
}

} // namespace

std::string render_comparison_metrics(
    const VariantMeasurement& baseline, const VariantMeasurement& candidate) {
  std::ostringstream table_html;
  table_html << "<div class=\"table-wrap\"><table><thead><tr><th>Metric</th><th>"
             << escape_html_text(baseline.variant_name) << "</th><th>"
             << escape_html_text(candidate.variant_name)
             << "</th><th>Change</th></tr></thead><tbody>";
  for (const MetricRow& metric_row : collect_metric_rows(baseline, candidate)) {
    table_html << "<tr><td>" << metric_row.metric_name << "</td><td class=\"number\">"
               << escape_html_text(format_metric(metric_row.baseline_value,
                                                 metric_row.is_duration))
               << "</td><td class=\"number\">"
               << escape_html_text(format_metric(metric_row.candidate_value,
                                                 metric_row.is_duration))
               << "</td><td class=\"number\">"
               << format_change(metric_row.baseline_value, metric_row.candidate_value)
               << "</td></tr>";
  }
  table_html << "</tbody></table></div><p class=\"note\">Lower is better for every"
             << " row; a negative change is an improvement. Each variant ran the"
             << " same four passes on the same materialized inputs. Timeline pass: "
             << escape_html_text(baseline.variant_name) << " timed "
             << granularity_name(baseline.gpu_timeline.granularity) << ", "
             << escape_html_text(candidate.variant_name) << " timed "
             << granularity_name(candidate.gpu_timeline.granularity)
             << (baseline.gpu_timeline.granularity == candidate.gpu_timeline.granularity
                     ? "."
                     : " — different granularities, so the GPU execution and"
                       " instrumented rows are not like for like.")
             << "</p>";
  return table_html.str();
}

namespace {

std::string render_variant_column(const ComparedVariant& compared_variant) {
  const VariantMeasurement& variant = compared_variant.variant_measurement;
  std::ostringstream column_html;
  column_html << "<div class=\"variant\"><h3>" << escape_html_text(variant.variant_name)
              << " <small>(" << variant.gpu_timeline.nodes.size()
              << " nodes) — <a href=\""
              << escape_html_text(compared_variant.full_report_file)
              << "\">full report</a></small></h3><div class=\"dag\">";
  if (compared_variant.plan_svg.empty()) {
    column_html << "<p class=\"note\">Graphviz unavailable: open the full report"
                << " for the DOT source.</p>";
  } else {
    column_html << compared_variant.plan_svg;
  }
  column_html << "</div></div>";
  return column_html.str();
}

struct PrimitiveTally {
  int node_count = 0;
  double compute_microseconds = 0.0;
};

// A stage-timed variant has no per-node GPU time, so its per-primitive time
// comes from the serial pass (each node alone) — labeled as such, never
// shown as zero compute.
std::map<std::string, PrimitiveTally> tally_primitives(const VariantMeasurement& variant) {
  const GpuTimeline& gpu_timeline = variant.gpu_timeline;
  const bool is_stage_timed = gpu_timeline.granularity == TimelineGranularity::per_stage;
  std::map<std::string, PrimitiveTally> op_name_to_tally;
  for (std::size_t position = 0; position < gpu_timeline.nodes.size(); ++position) {
    const NodeTimeline& node_timeline = gpu_timeline.nodes[position];
    PrimitiveTally& tally = op_name_to_tally[node_timeline.node->op_name];
    ++tally.node_count;
    if (!is_stage_timed) {
      tally.compute_microseconds += node_timeline.breakdown.compute_microseconds;
    } else if (const NodeTelemetry* serial_node = find_attributed_node(
                   variant.serial_telemetry, position, node_timeline.node->op_name)) {
      tally.compute_microseconds += serial_node->duration_microseconds;
    }
  }
  return op_name_to_tally;
}

const char* primitive_time_label(const VariantMeasurement& variant) {
  return variant.gpu_timeline.granularity == TimelineGranularity::per_stage
      ? " serial Σ (stage-timed)"
      : " compute";
}

void append_tally_cells(
    std::ostringstream& row_html,
    const std::map<std::string, PrimitiveTally>& op_name_to_tally,
    const std::string& op_name) {
  const auto tally_entry = op_name_to_tally.find(op_name);
  if (tally_entry == op_name_to_tally.end()) {
    row_html << "<td class=\"number\" data-sort-value=\"0\">—</td>"
             << "<td class=\"number\" data-sort-value=\"0\">—</td>";
    return;
  }
  row_html << "<td class=\"number\" data-sort-value=\""
           << tally_entry->second.node_count << "\">"
           << tally_entry->second.node_count << "</td><td class=\"number\""
           << " data-sort-value=\"" << tally_entry->second.compute_microseconds
           << "\">" << escape_html_text(format_microseconds(
                            tally_entry->second.compute_microseconds))
           << "</td>";
}

// Spark's "planned vs executed": which primitives each variant actually ran,
// and how much measured GPU time each kind took. Fused kernels exist only on
// the compiled side; the primitives they absorbed shrink or vanish.
std::string render_primitive_mix(
    const VariantMeasurement& baseline, const VariantMeasurement& candidate) {
  const auto baseline_tallies = tally_primitives(baseline);
  const auto candidate_tallies = tally_primitives(candidate);
  std::map<std::string, bool> all_op_names;
  for (const auto& [op_name, tally] : baseline_tallies) {
    all_op_names[op_name] = true;
  }
  for (const auto& [op_name, tally] : candidate_tallies) {
    all_op_names[op_name] = true;
  }
  std::ostringstream table_html;
  table_html << "<div class=\"table-wrap\"><table class=\"sortable\"><thead><tr>"
             << "<th>Primitive</th><th>Fuses</th><th>" << escape_html_text(baseline.variant_name)
             << " nodes</th><th>" << escape_html_text(baseline.variant_name)
             << primitive_time_label(baseline) << "</th><th>"
             << escape_html_text(candidate.variant_name) << " nodes</th><th>"
             << escape_html_text(candidate.variant_name) << primitive_time_label(candidate)
             << "</th></tr></thead><tbody>";
  for (const auto& [op_name, is_present] : all_op_names) {
    table_html << "<tr><td>" << escape_html_text(op_name) << "</td><td>"
               << escape_html_text(describe_fused_primitives(op_name)) << "</td>";
    append_tally_cells(table_html, baseline_tallies, op_name);
    append_tally_cells(table_html, candidate_tallies, op_name);
    table_html << "</tr>";
  }
  table_html << "</tbody></table></div>";
  return table_html.str();
}

std::string render_shared_scale_timelines(
    const VariantMeasurement& baseline, const VariantMeasurement& candidate) {
  const double shared_scale_microseconds =
      std::max(baseline.gpu_timeline.timeline_wall_microseconds,
               candidate.gpu_timeline.timeline_wall_microseconds);
  std::ostringstream timelines_html;
  timelines_html << render_slice_legend() << "<p class=\"note\">Both timeline"
                 << " passes on one time axis, so lengths compare directly.</p>";
  for (const VariantMeasurement* variant : {&baseline, &candidate}) {
    timelines_html << "<h3>" << escape_html_text(variant->variant_name)
                   << "</h3><div class=\"gantt-wrap\">"
                   << render_gantt_svg(variant->gpu_timeline, shared_scale_microseconds)
                   << "</div>";
  }
  return timelines_html.str();
}

} // namespace

std::string describe_fused_primitives(const std::string& compiled_op_name) {
  if (compiled_op_name.rfind(COMPILED_PREFIX, 0) != 0 ||
      compiled_op_name.size() == COMPILED_PREFIX.size()) {
    return "";
  }
  std::string fused_description;
  for (std::size_t letter_position = COMPILED_PREFIX.size();
       letter_position < compiled_op_name.size(); ++letter_position) {
    const char letter = compiled_op_name[letter_position];
    const bool starts_new_primitive =
        std::isupper(static_cast<unsigned char>(letter)) &&
        letter_position > COMPILED_PREFIX.size() &&
        !std::isupper(static_cast<unsigned char>(compiled_op_name[letter_position - 1]));
    if (starts_new_primitive) {
      fused_description += " · ";
    }
    fused_description += letter;
  }
  return fused_description;
}

std::string build_comparison_document(
    const std::string& page_title,
    const ComparedVariant& baseline_variant,
    const ComparedVariant& candidate_variant) {
  const VariantMeasurement& baseline = baseline_variant.variant_measurement;
  const VariantMeasurement& candidate = candidate_variant.variant_measurement;
  std::ostringstream page_html;
  page_html << "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            << "<title>" << escape_html_text(page_title) << "</title><style>"
            << report_stylesheet() << "</style></head><body><header><h1>"
            << escape_html_text(page_title) << "</h1><p>Same job, two plans: "
            << escape_html_text(baseline.variant_name) << " vs "
            << escape_html_text(candidate.variant_name) << "</p></header>"
            << "<section id=\"compare\"><h2>Job metrics</h2>"
            << render_comparison_metrics(baseline, candidate) << "</section>"
            << "<section id=\"plans\"><h2>Plans side by side</h2><div class=\"variant-grid\">"
            << render_variant_column(baseline_variant)
            << render_variant_column(candidate_variant) << "</div></section>"
            << "<section id=\"timelines\"><h2>Timelines (shared scale)</h2>"
            << render_shared_scale_timelines(baseline, candidate) << "</section>"
            << "<section id=\"primitives\"><h2>Primitive mix (planned vs executed)</h2>"
            << render_primitive_mix(baseline, candidate) << "</section>"
            << "<script>" << report_sort_script() << "</script></body></html>\n";
  return page_html.str();
}

} // namespace mlx_plan
