#include "report_memory_section.h"

#include "text_format.h"

#include <algorithm>
#include <sstream>

namespace mlx_perf {
namespace {

constexpr int TOP_MEMORY_NODE_COUNT = 15;
constexpr double CHART_WIDTH_PIXELS = 1000.0;
constexpr double CHART_HEIGHT_PIXELS = 180.0;
constexpr double CHART_LEFT_PIXELS = 70.0;
constexpr double CHART_TOP_PIXELS = 10.0;
constexpr double CHART_BOTTOM_PIXELS = 24.0;

bool has_more_memory_added(const NodeTelemetry* left, const NodeTelemetry* right) {
  return left->memory_added_bytes > right->memory_added_bytes;
}

std::vector<const NodeTelemetry*> rank_by_memory_added(const RunTelemetry& serial_telemetry) {
  std::vector<const NodeTelemetry*> ranked_nodes;
  for (const NodeTelemetry& node_telemetry : serial_telemetry.attributed_nodes) {
    ranked_nodes.push_back(&node_telemetry);
  }
  std::stable_sort(ranked_nodes.begin(), ranked_nodes.end(), has_more_memory_added);
  if (ranked_nodes.size() > static_cast<std::size_t>(TOP_MEMORY_NODE_COUNT)) {
    ranked_nodes.resize(TOP_MEMORY_NODE_COUNT);
  }
  return ranked_nodes;
}

std::string render_polyline(const std::vector<MemorySample>& memory_samples,
                            std::size_t MemorySample::*byte_field,
                            double max_bytes, const char* css_class) {
  const double first_seconds = memory_samples.front().seconds_since_start;
  const double span_seconds =
      std::max(1.0, memory_samples.back().seconds_since_start - first_seconds);
  const double plot_height = CHART_HEIGHT_PIXELS - CHART_TOP_PIXELS - CHART_BOTTOM_PIXELS;
  std::ostringstream points_text;
  for (const MemorySample& memory_sample : memory_samples) {
    const double sample_x = CHART_LEFT_PIXELS +
        (CHART_WIDTH_PIXELS - CHART_LEFT_PIXELS) *
            (memory_sample.seconds_since_start - first_seconds) / span_seconds;
    const double sample_y = CHART_TOP_PIXELS + plot_height *
        (1.0 - static_cast<double>(memory_sample.*byte_field) / max_bytes);
    points_text << sample_x << "," << sample_y << " ";
  }
  return "<polyline class=\"" + std::string(css_class) +
      "\" fill=\"none\" stroke-width=\"1.6\" points=\"" + points_text.str() + "\"/>";
}

} // namespace

std::string render_memory_section(const VariantMeasurement& variant_measurement) {
  const RunTelemetry& serial_telemetry = variant_measurement.serial_telemetry;
  const auto ranked_nodes = rank_by_memory_added(serial_telemetry);
  const std::size_t largest_added_bytes =
      ranked_nodes.empty() ? 0 : ranked_nodes.front()->memory_added_bytes;
  std::ostringstream section_html;
  section_html << "<p class=\"note\">Peak memory during the serial pass: <b>"
               << format_bytes(serial_telemetry.run_peak_memory_bytes)
               << "</b>. Nodes ranked by active memory added when run alone"
               << " (approximate: the capture keeps intermediates alive and MLX"
               << " reuses buffers).</p><div class=\"table-wrap\"><table class=\"sortable\">"
               << "<thead><tr><th>Node</th><th>Op</th><th>Output</th><th>Mem added</th>"
               << "<th>Share of largest</th></tr></thead><tbody>";
  for (const NodeTelemetry* node_telemetry : ranked_nodes) {
    const double share_percent = largest_added_bytes > 0
        ? 100.0 * static_cast<double>(node_telemetry->memory_added_bytes) /
              static_cast<double>(largest_added_bytes)
        : 0.0;
    section_html << "<tr><td>" << escape_html_text(node_telemetry->node->label)
                 << "</td><td>" << escape_html_text(node_telemetry->node->op_name)
                 << "</td><td class=\"number\">" << format_bytes(node_telemetry->output_bytes)
                 << "</td><td class=\"number\" data-sort-value=\""
                 << node_telemetry->memory_added_bytes << "\">"
                 << format_bytes(node_telemetry->memory_added_bytes)
                 << "</td><td><div class=\"stacked-bar\"><span class=\"slice-queued\""
                 << " style=\"width:" << share_percent << "%\"></span></div></td></tr>";
  }
  section_html << "</tbody></table></div>";
  return section_html.str();
}

std::string render_memory_timeline(const std::vector<MemorySample>& memory_samples) {
  if (memory_samples.size() < 2) {
    return "<p class=\"note\">Collecting memory samples (one per second)…</p>";
  }
  std::size_t max_bytes = 1;
  for (const MemorySample& memory_sample : memory_samples) {
    max_bytes = std::max({max_bytes, memory_sample.active_bytes,
                          memory_sample.peak_bytes, memory_sample.cache_bytes});
  }
  const MemorySample& latest_sample = memory_samples.back();
  std::ostringstream chart_html;
  chart_html << "<div class=\"legend\"><span><i class=\"swatch slice-compute\"></i>active "
             << format_bytes(latest_sample.active_bytes)
             << "</span><span><i class=\"swatch slice-scheduling\"></i>peak "
             << format_bytes(latest_sample.peak_bytes)
             << "</span><span><i class=\"swatch slice-result\"></i>cache "
             << format_bytes(latest_sample.cache_bytes) << "</span></div>"
             << "<div class=\"gantt-wrap\"><svg class=\"gantt memory-chart\" width=\""
             << CHART_WIDTH_PIXELS << "\" height=\"" << CHART_HEIGHT_PIXELS
             << "\" role=\"img\" aria-label=\"Process memory over time\">"
             << "<text class=\"axis-text\" x=\"4\" y=\"" << CHART_TOP_PIXELS + 8 << "\">"
             << format_bytes(max_bytes) << "</text><text class=\"axis-text\" x=\"4\" y=\""
             << CHART_HEIGHT_PIXELS - CHART_BOTTOM_PIXELS << "\">0 B</text>"
             << render_polyline(memory_samples, &MemorySample::cache_bytes,
                                static_cast<double>(max_bytes), "line-result")
             << render_polyline(memory_samples, &MemorySample::peak_bytes,
                                static_cast<double>(max_bytes), "line-scheduling")
             << render_polyline(memory_samples, &MemorySample::active_bytes,
                                static_cast<double>(max_bytes), "line-compute")
             << "<text class=\"axis-text\" x=\"" << CHART_LEFT_PIXELS << "\" y=\""
             << CHART_HEIGHT_PIXELS - 6 << "\">last "
             << static_cast<int>(latest_sample.seconds_since_start -
                                 memory_samples.front().seconds_since_start)
             << " s</text></svg></div>";
  return chart_html.str();
}

} // namespace mlx_perf
