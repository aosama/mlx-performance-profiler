#include "report_stage_table_section.h"

#include "report_components.h"
#include "text_format.h"

#include <algorithm>
#include <map>
#include <sstream>

namespace mlx_perf {
namespace {

constexpr int SHOWN_OP_COUNT = 4;

bool is_more_frequent(const std::pair<std::string, int>& left_op,
                      const std::pair<std::string, int>& right_op) {
  return left_op.second > right_op.second;
}

// Summed over the stage's members; incomplete when any member failed the
// serial join, which the row then says instead of showing a partial sum.
struct SerialTotals {
  double duration_microseconds = 0.0;
  size_t memory_added_bytes = 0;
  bool is_complete = true;
};

SerialTotals sum_serial_members(const ReportInputs& report_inputs,
                                const StageTimeline& stage_timeline) {
  SerialTotals serial_totals;
  for (const int member_position : stage_timeline.node_positions) {
    const GraphNode& member_node = *report_inputs.gpu_timeline.nodes[member_position].node;
    const NodeTelemetry* serial_node = find_attributed_node(
        report_inputs.serial_telemetry, member_position, member_node.op_name);
    if (serial_node == nullptr) {
      serial_totals.is_complete = false;
      continue;
    }
    serial_totals.duration_microseconds += serial_node->duration_microseconds;
    serial_totals.memory_added_bytes += serial_node->memory_added_bytes;
  }
  return serial_totals;
}

size_t sum_output_bytes(const GpuTimeline& gpu_timeline, const StageTimeline& stage_timeline) {
  size_t output_bytes = 0;
  for (const int member_position : stage_timeline.node_positions) {
    output_bytes += gpu_timeline.nodes[member_position].node->output_bytes;
  }
  return output_bytes;
}

std::string render_header_row() {
  return "<thead><tr><th>#</th><th>Stage</th><th>Stream</th><th>Nodes</th>"
         "<th>Ops inside</th><th title=\"Reads data produced on another stream in an"
         " earlier stage\">Join</th><th>Output</th>" +
      render_slice_header_cells() +
      "<th>Wall</th><th title=\"Members timed alone in the serial pass, summed\">"
      "Serial Σ</th><th title=\"Active memory added by the members in the serial"
      " pass, summed (approximate: buffers are reused)\">Mem added Σ</th>"
      "<th>Class</th><th>Latency breakdown</th></tr></thead>";
}

std::string render_stage_row(std::size_t stage_task_position,
                             const StageTimeline& stage_timeline,
                             const ReportInputs& report_inputs) {
  const GpuTimeline& gpu_timeline = report_inputs.gpu_timeline;
  std::ostringstream row_html;
  row_html << "<tr>"
           << render_number_cell(static_cast<double>(stage_task_position + 1),
                                 std::to_string(stage_task_position + 1))
           << "<td>" << escape_html_text(stage_timeline.stage_name) << "</td>"
           << render_number_cell(stage_timeline.stream_index,
                                 std::to_string(stage_timeline.stream_index))
           << render_number_cell(static_cast<double>(stage_timeline.node_positions.size()),
                                 std::to_string(stage_timeline.node_positions.size()))
           << "<td>"
           << escape_html_text(summarize_op_mix(gpu_timeline, stage_timeline.node_positions,
                                                SHOWN_OP_COUNT))
           << "</td><td>" << (stage_timeline.has_cross_stream_input ? "yes" : "") << "</td>";
  const size_t output_bytes = sum_output_bytes(gpu_timeline, stage_timeline);
  row_html << render_number_cell(static_cast<double>(output_bytes), format_bytes(output_bytes))
           << render_slice_cells(stage_timeline.breakdown)
           << render_duration_cell(stage_timeline.breakdown.wall_microseconds);
  const SerialTotals serial_totals = sum_serial_members(report_inputs, stage_timeline);
  if (serial_totals.is_complete) {
    row_html << render_duration_cell(serial_totals.duration_microseconds)
             << render_number_cell(static_cast<double>(serial_totals.memory_added_bytes),
                                   format_bytes(serial_totals.memory_added_bytes));
  } else {
    row_html << render_number_cell(0, "—") << render_number_cell(0, "—");
  }
  row_html << "<td>" << render_class_badge(stage_timeline.breakdown.classification)
           << "</td><td>" << render_stacked_bar(stage_timeline.breakdown) << "</td></tr>";
  return row_html.str();
}

} // namespace

std::string summarize_op_mix(const GpuTimeline& gpu_timeline,
                             const std::vector<int>& member_positions,
                             int shown_op_count) {
  std::map<std::string, int> op_name_to_count;
  for (const int member_position : member_positions) {
    ++op_name_to_count[gpu_timeline.nodes[member_position].node->op_name];
  }
  std::vector<std::pair<std::string, int>> ranked_ops(op_name_to_count.begin(),
                                                      op_name_to_count.end());
  // stable_sort keeps the map's alphabetical order among equal counts.
  std::stable_sort(ranked_ops.begin(), ranked_ops.end(), is_more_frequent);
  std::string op_mix;
  const int shown_count = std::min(shown_op_count, static_cast<int>(ranked_ops.size()));
  for (int rank = 0; rank < shown_count; ++rank) {
    op_mix += (rank > 0 ? ", " : "") + ranked_ops[rank].first + " ×" +
        std::to_string(ranked_ops[rank].second);
  }
  if (shown_count < static_cast<int>(ranked_ops.size())) {
    op_mix += ", …";
  }
  return op_mix;
}

std::string render_stage_table_section(const ReportInputs& report_inputs) {
  std::ostringstream section_html;
  section_html << "<section id=\"tasks\"><h2>Tasks (one per stage and stream)</h2>"
               << render_slice_legend()
               << "<p class=\"note\">This job was timed per STAGE: one async_eval per"
               << " stage, so the instrumentation stays close to a plain eval. A row"
               << " covers every member of the stage on one stream; members finish"
               << " together, so per-node waiting is not observable here. Compute is"
               << " the stage's GPU window (first command buffer start → end of its"
               << " last buffer, bounded by a marker buffer) and includes any GPU gap"
               << " while the host was still encoding the stage. Serial Σ sums the"
               << " members timed alone. Click a header to sort.</p>"
               << "<div class=\"table-wrap\"><table class=\"sortable\">"
               << render_header_row() << "<tbody>";
  const auto& stage_timelines = report_inputs.gpu_timeline.stages;
  for (std::size_t stage_task_position = 0; stage_task_position < stage_timelines.size();
       ++stage_task_position) {
    section_html << render_stage_row(stage_task_position, stage_timelines[stage_task_position],
                                     report_inputs);
  }
  section_html << "</tbody></table></div></section>";
  return section_html.str();
}

} // namespace mlx_perf
