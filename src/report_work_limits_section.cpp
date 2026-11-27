#include "report_work_limits_section.h"

#include "gpu_backend_vocabulary.h"
#include "report_components.h"
#include "text_format.h"
#include "timed_unit_work.h"

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <vector>

namespace mlx_perf {
namespace {

// One table row: a timed unit that moved data.
struct WorkRow {
  std::string unit_name;
  std::string operation_description;
  double gpu_compute_microseconds = 0.0;
  NodeWork unit_work;
  WorkJudgement work_judgement;
};

std::string describe_node_operation(const GraphNode& graph_node) {
  return graph_node.kernel_name.empty() ? graph_node.op_name : graph_node.kernel_name;
}

std::vector<WorkRow> collect_node_rows(const ReportInputs& report_inputs) {
  std::vector<WorkRow> work_rows;
  const GpuTimeline& gpu_timeline = report_inputs.gpu_timeline;
  for (std::size_t position = 0; position < gpu_timeline.nodes.size(); ++position) {
    const NodeWork node_work = work_of_node(gpu_timeline, position, report_inputs.serial_telemetry);
    if (node_work.bytes_read + node_work.bytes_written == 0) {
      continue;
    }
    const NodeTimeline& node_timeline = gpu_timeline.nodes[position];
    work_rows.push_back({node_timeline.node->label, describe_node_operation(*node_timeline.node),
                         node_timeline.breakdown.compute_microseconds, node_work,
                         judge_node(gpu_timeline, position, report_inputs.serial_telemetry,
                                    report_inputs.machine_ceilings)});
  }
  return work_rows;
}

std::vector<WorkRow> collect_stage_rows(const ReportInputs& report_inputs) {
  std::vector<WorkRow> work_rows;
  const GpuTimeline& gpu_timeline = report_inputs.gpu_timeline;
  for (const StageTimeline& stage_task : gpu_timeline.stages) {
    const NodeWork stage_work = work_of_stage(gpu_timeline, stage_task, report_inputs.serial_telemetry);
    if (stage_work.bytes_read + stage_work.bytes_written == 0) {
      continue;
    }
    const std::size_t member_count = stage_task.node_positions.size();
    work_rows.push_back({stage_task.stage_name,
                         std::to_string(member_count) + (member_count == 1 ? " node" : " nodes") +
                             ", stream " + std::to_string(stage_task.stream_index),
                         stage_task.breakdown.compute_microseconds, stage_work,
                         judge_stage(gpu_timeline, stage_task, report_inputs.serial_telemetry,
                                     report_inputs.machine_ceilings)});
  }
  return work_rows;
}

std::string format_share(double share) {
  char formatted_share[24];
  std::snprintf(formatted_share, sizeof(formatted_share), "%.0f%%", share * 100.0);
  return formatted_share;
}

// Two bars, data movement over arithmetic, each the share of the unit's GPU
// time that need would take; capped at the cell's width when a kernel beat
// the calibration, with the true share in the hover text.
std::string render_need_bars(const WorkJudgement& work_judgement) {
  const double data_width = std::min(work_judgement.data_movement_share, 1.0) * 100.0;
  const double arithmetic_width = std::min(work_judgement.arithmetic_share, 1.0) * 100.0;
  std::ostringstream bars_html;
  bars_html << "<div class=\"need-bars\" title=\"Data movement " << format_share(work_judgement.data_movement_share)
            << ", arithmetic "
            << (work_judgement.has_arithmetic_time ? format_share(work_judgement.arithmetic_share) : "not counted")
            << "\"><div class=\"need-bar need-data\" style=\"width:" << data_width << "%\"></div>"
            << "<div class=\"need-bar need-arithmetic\" style=\"width:" << arithmetic_width << "%\"></div></div>";
  return bars_html.str();
}

std::string render_limit_badge(WorkLimit work_limit) {
  switch (work_limit) {
    case WorkLimit::data_movement:
      return "<span class=\"limit-badge limit-data\">data movement</span>";
    case WorkLimit::arithmetic:
      return "<span class=\"limit-badge limit-arithmetic\">arithmetic</span>";
    case WorkLimit::neither:
      return "<span class=\"limit-badge limit-neither\">neither</span>";
    case WorkLimit::not_judged:
      return "—";
  }
  return "—";
}

std::string render_share_cell(bool has_share, double share) {
  return has_share ? render_number_cell(share, format_share(share))
                   : "<td class=\"number\" data-sort-value=\"-1\">—</td>";
}

std::string render_work_row(std::size_t row_number, const WorkRow& work_row) {
  const NodeWork& unit_work = work_row.unit_work;
  const WorkJudgement& judgement = work_row.work_judgement;
  std::ostringstream row_html;
  row_html << "<tr>" << render_number_cell(static_cast<double>(row_number), std::to_string(row_number))
           << "<td>" << escape_html_text(work_row.unit_name) << "</td><td>"
           << escape_html_text(work_row.operation_description) << "</td>"
           << render_duration_cell(work_row.gpu_compute_microseconds)
           << render_number_cell(static_cast<double>(unit_work.bytes_read + unit_work.bytes_written),
                                 format_bytes(unit_work.bytes_read + unit_work.bytes_written))
           << (unit_work.is_arithmetic_counted
                   ? render_number_cell(unit_work.arithmetic_operations,
                                        format_operation_count(unit_work.arithmetic_operations))
                   : "<td class=\"number\" data-sort-value=\"-1\">not counted</td>")
           << render_share_cell(judgement.has_shares, judgement.data_movement_share)
           << render_share_cell(judgement.has_shares && judgement.has_arithmetic_time, judgement.arithmetic_share)
           << "<td>" << (judgement.has_shares ? render_need_bars(judgement) : "") << "</td><td>"
           << render_limit_badge(judgement.limited_by) << "</td></tr>";
  return row_html.str();
}

std::string render_card(const std::string& card_value, const std::string& card_label,
                        const std::string& card_detail) {
  return "<div class=\"card\"><div class=\"card-value\">" + escape_html_text(card_value) +
         "</div><div class=\"card-label\">" + escape_html_text(card_label) +
         "</div><div class=\"card-detail\">" + escape_html_text(card_detail) + "</div></div>";
}

std::string render_ceiling_cards(const MachineCeilings& machine_ceilings,
                                 const std::vector<WorkRow>& work_rows) {
  double judged_microseconds = 0.0;
  double data_movement_microseconds = 0.0;
  double arithmetic_microseconds = 0.0;
  for (const WorkRow& work_row : work_rows) {
    if (work_row.work_judgement.has_shares) {
      judged_microseconds += work_row.gpu_compute_microseconds;
      data_movement_microseconds += work_row.work_judgement.data_movement_microseconds;
      arithmetic_microseconds += work_row.work_judgement.arithmetic_microseconds;
    }
  }
  return "<div class=\"cards\">" +
      render_card(format_bytes_per_second(machine_ceilings.memory_read_bytes_per_second),
                  "Reading ceiling", std::string(gpu_vocabulary::MEMORY_NAME) + ", measured on this machine") +
      render_card(format_operations_per_second(machine_ceilings.arithmetic_operations_per_second),
                  "Arithmetic ceiling", "a " + machine_ceilings.arithmetic_dtype + " matrix multiply") +
      render_card(format_microseconds(judged_microseconds), "GPU time judged", "units with a GPU window") +
      render_card(format_microseconds(data_movement_microseconds), "Data movement at the ceiling",
                  "what their bytes need") +
      render_card(format_microseconds(arithmetic_microseconds), "Arithmetic at the ceiling",
                  "what their counted arithmetic needs") +
      "</div>";
}

std::string render_header_row() {
  return "<thead><tr><th>#</th><th>Unit</th><th>Operation</th><th>GPU time</th>"
         "<th title=\"Bytes read and written, worked out from the graph\">"
         "Bytes moved</th><th title=\"Two per multiply-add; counted for matrix multiplies and"
         " fused attention\">Arithmetic</th><th title=\"Time the bytes need at the reading"
         " ceiling, over the GPU time\">Data share</th><th title=\"Time the arithmetic needs at the"
         " arithmetic ceiling, over the GPU time\">Arithmetic share</th><th>Needs</th>"
         "<th>Limited by</th></tr></thead>";
}

std::string render_section_introduction() {
  const std::string memory_name = gpu_vocabulary::MEMORY_NAME;
  return "<p class=\"note\">A kernel moves its data out of " + memory_name +
         " while it runs, overlapped with its arithmetic, so no clock splits a kernel's time between"
         " the two. What can be said is how much of its GPU time each would need on its own, at"
         " this machine's measured ceilings. The larger share names the limit when it fills at"
         " least half the time; <b>neither</b> means the GPU was not kept busy, which is often where"
         " a kernel can be made faster. Bytes count every input once and every output, as traffic"
         " to " + memory_name + "; a gathered multiply counts only the experts its indices used. A"
         " share above 100% means the kernel beat the calibration kernel.</p>";
}

} // namespace

std::string render_work_limits_section(const ReportInputs& report_inputs) {
  const bool is_stage_timed = report_inputs.gpu_timeline.granularity == TimelineGranularity::per_stage;
  std::vector<WorkRow> work_rows =
      is_stage_timed ? collect_stage_rows(report_inputs) : collect_node_rows(report_inputs);
  std::ostringstream section_html;
  section_html << "<section id=\"data-and-arithmetic\"><h2>Data and arithmetic</h2>"
               << render_section_introduction();
  if (!report_inputs.machine_ceilings.is_measured) {
    section_html << "<p class=\"note\">This machine's ceilings were not measured (the GPU gave no"
                 << " execution window), so no unit is judged.</p></section>";
    return section_html.str();
  }
  section_html << render_ceiling_cards(report_inputs.machine_ceilings, work_rows)
               << "<div class=\"table-wrap\"><table class=\"sortable\">" << render_header_row() << "<tbody>";
  for (std::size_t row_index = 0; row_index < work_rows.size(); ++row_index) {
    section_html << render_work_row(row_index + 1, work_rows[row_index]);
  }
  section_html << "</tbody></table></div></section>";
  return section_html.str();
}

} // namespace mlx_perf
