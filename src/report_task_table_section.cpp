#include "report_task_table_section.h"

#include "kernel_launch_summary.h"
#include "latency_slices.h"
#include "report_components.h"
#include "text_format.h"

#include <algorithm>
#include <sstream>

namespace mlx_perf {
namespace {

void append_number_cell(
    std::ostringstream& row_html, double sort_value, const std::string& shown) {
  row_html << render_number_cell(sort_value, shown);
}

void append_duration_cell(std::ostringstream& row_html, double microseconds) {
  row_html << render_duration_cell(microseconds);
}

std::string render_header_row() {
  std::ostringstream header_html;
  header_html << "<thead><tr><th>#</th><th>Node</th><th>Op</th><th>Stream</th>"
              << "<th>Join</th><th>Shape</th><th>Dtype</th><th>Output</th>";
  header_html << render_slice_header_cells();
  header_html << "<th>Wall</th><th title=\"Same node timed alone in the serial"
              << " pass\">Serial</th><th title=\"Active memory added by this"
              << " node in the serial pass (approximate: buffers are reused)\">"
              << "Mem added</th><th title=\"Kernel launches in the serial pass: threadgroups,"
              << " and how many GPU cores they can give work to\">Launches</th>"
              << "<th>Class</th><th>Latency breakdown</th>"
              << "</tr></thead>";
  return header_html.str();
}

// A node with a launch that cannot fill the GPU is marked, because that is
// the reason a kernel can be slow that no timing shows: idle cores.
std::string render_launches_cell(const NodeTelemetry* serial_node, int gpu_core_count) {
  if (serial_node == nullptr || serial_node->kernel_dispatches.empty()) {
    return "<td class=\"number\" data-sort-value=\"0\">—</td>";
  }
  std::uint64_t fewest_threadgroups = count_threadgroups(serial_node->kernel_dispatches.front());
  for (const KernelDispatch& kernel_dispatch : serial_node->kernel_dispatches) {
    fewest_threadgroups = std::min(fewest_threadgroups, count_threadgroups(kernel_dispatch));
  }
  const bool is_unable_to_fill_gpu =
      gpu_core_count > 0 && !can_fill_gpu(fewest_threadgroups, gpu_core_count);
  std::ostringstream cell_html;
  cell_html << "<td class=\"number" << (is_unable_to_fill_gpu ? " underfills-gpu" : "")
            << "\" data-sort-value=\"" << fewest_threadgroups << "\">"
            << (is_unable_to_fill_gpu ? "⚠ " : "")
            << escape_html_text(describe_node_launches(serial_node->kernel_dispatches, gpu_core_count))
            << "</td>";
  return cell_html.str();
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
  row_html << render_slice_cells(breakdown);
  append_duration_cell(row_html, breakdown.wall_microseconds);
  const NodeTelemetry* serial_node =
      find_attributed_node(serial_telemetry, position, graph_node.op_name);
  if (serial_node != nullptr) {
    append_duration_cell(row_html, serial_node->duration_microseconds);
    append_number_cell(row_html,
                       static_cast<double>(serial_node->memory_added_bytes),
                       format_bytes(serial_node->memory_added_bytes));
  } else {
    row_html << "<td class=\"number\" data-sort-value=\"0\">—</td>"
             << "<td class=\"number\" data-sort-value=\"0\">—</td>";
  }
  row_html << render_launches_cell(serial_node, serial_telemetry.gpu_device_facts.gpu_core_count);
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
               << " is the same node timed alone, for comparison. Launches are"
               << " its kernel launches in that pass; ⚠ marks a launch with"
               << " fewer threadgroups than the GPU has cores, which leaves"
               << " cores idle.</p>"
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

} // namespace mlx_perf
