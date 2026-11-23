#include "report_task_table_section.h"

#include "latency_slices.h"
#include "report_components.h"
#include "text_format.h"

#include <sstream>

namespace mlx_plan {
namespace {

void append_number_cell(
    std::ostringstream& row_html, double sort_value, const std::string& shown) {
  row_html << "<td class=\"number\" data-sort-value=\"" << sort_value << "\">"
           << escape_html_text(shown) << "</td>";
}

void append_duration_cell(std::ostringstream& row_html, double microseconds) {
  append_number_cell(row_html, microseconds, format_microseconds(microseconds));
}

// The serial pass ran an identical graph, so topological position joins the
// two; the op name guards the join, and a mismatch shows "—", never a
// number borrowed from a different node.
const NodeTelemetry* find_serial_node(
    const RunTelemetry& serial_telemetry,
    std::size_t position,
    const GraphNode& timeline_node) {
  if (position >= serial_telemetry.attributed_nodes.size()) {
    return nullptr;
  }
  const NodeTelemetry& serial_node = serial_telemetry.attributed_nodes[position];
  return serial_node.node->op_name == timeline_node.op_name ? &serial_node
                                                            : nullptr;
}

std::string render_header_row() {
  std::ostringstream header_html;
  header_html << "<thead><tr><th>#</th><th>Node</th><th>Op</th><th>Stream</th>"
              << "<th>Join</th><th>Shape</th><th>Dtype</th><th>Output</th>";
  for (const LatencySlice latency_slice : CAUSAL_SLICE_ORDER) {
    const SliceStyle& style = slice_style(latency_slice);
    header_html << "<th title=\"" << escape_html_text(style.meaning) << "\">"
                << style.short_label << "</th>";
  }
  header_html << "<th>Wall</th><th title=\"Same node timed alone in the serial"
              << " pass\">Serial</th><th title=\"Active memory added by this"
              << " node in the serial pass (approximate: buffers are reused)\">"
              << "Mem added</th><th>Class</th><th>Latency breakdown</th>"
              << "</tr></thead>";
  return header_html.str();
}

std::string render_node_row(
    std::size_t position,
    const NodeTimeline& node_timeline,
    const RunTelemetry& serial_telemetry) {
  const GraphNode& graph_node = *node_timeline.node;
  const LatencyBreakdown& breakdown = node_timeline.breakdown;
  std::ostringstream row_html;
  row_html << "<tr><td class=\"number\" data-sort-value=\"" << position + 1
           << "\">" << position + 1 << "</td><td>"
           << escape_html_text(graph_node.label) << "</td><td>"
           << escape_html_text(graph_node.op_name) << "</td>";
  append_number_cell(
      row_html, graph_node.stream_index, std::to_string(graph_node.stream_index));
  row_html << "<td>" << (node_timeline.has_cross_stream_input ? "yes" : "")
           << "</td><td>" << escape_html_text(graph_node.shape_string)
           << "</td><td>" << escape_html_text(graph_node.dtype_string) << "</td>";
  append_number_cell(row_html, static_cast<double>(graph_node.output_bytes),
                     format_bytes(graph_node.output_bytes));
  for (const LatencySlice latency_slice : CAUSAL_SLICE_ORDER) {
    const double slice_duration = slice_microseconds(breakdown, latency_slice);
    const bool is_residual_compute =
        latency_slice == LatencySlice::compute && !breakdown.is_compute_measured;
    append_number_cell(row_html, slice_duration,
        (is_residual_compute ? "~" : "") + format_microseconds(slice_duration));
  }
  append_duration_cell(row_html, breakdown.wall_microseconds);
  const NodeTelemetry* serial_node =
      find_serial_node(serial_telemetry, position, graph_node);
  if (serial_node != nullptr) {
    append_duration_cell(row_html, serial_node->duration_microseconds);
    append_number_cell(row_html,
                       static_cast<double>(serial_node->memory_added_bytes),
                       format_bytes(serial_node->memory_added_bytes));
  } else {
    row_html << "<td class=\"number\" data-sort-value=\"0\">—</td>"
             << "<td class=\"number\" data-sort-value=\"0\">—</td>";
  }
  row_html << "<td>" << render_class_badge(breakdown.classification)
           << "</td><td>" << render_stacked_bar(breakdown) << "</td></tr>";
  return row_html.str();
}

} // namespace

std::string render_task_table_section(const ReportInputs& report_inputs) {
  std::ostringstream section_html;
  section_html << "<section id=\"tasks\"><h2>Tasks (one per executed node)</h2>"
               << render_slice_legend()
               << "<p class=\"note\">Click a header to sort. Slices are in"
               << " causal order and sum to Wall. \"~\" marks compute that is"
               << " a host residual (no GPU window, e.g. CPU streams). Serial"
               << " is the same node timed alone, for comparison.</p>"
               << "<div class=\"table-wrap\"><table class=\"sortable\">"
               << render_header_row() << "<tbody>";
  const auto& node_timelines = report_inputs.gpu_timeline.nodes;
  for (std::size_t position = 0; position < node_timelines.size(); ++position) {
    section_html << render_node_row(
        position, node_timelines[position], report_inputs.serial_telemetry);
  }
  section_html << "</tbody></table></div></section>";
  return section_html.str();
}

} // namespace mlx_plan
