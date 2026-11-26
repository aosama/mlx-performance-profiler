#include "dot_writer.h"

#include "text_format.h"

#include <algorithm>
#include <map>
#include <sstream>
#include <unordered_map>

namespace mlx_perf {
namespace {

// Spark colors stages; here one hue per stream index. Chosen to stay legible
// with black text at small SVG sizes.
const char* stream_fill_color(int stream_index) {
  static const char* fill_colors[] = {
      "#cfe2ff", // blue — main stream
      "#d8f0d8", // green
      "#fde8cf", // orange
      "#e8d8f0", // purple
      "#cfe8e8", // teal
  };
  return fill_colors[stream_index % 5];
}

struct StageIndexMatcher {
  int stage_index;
  bool operator()(const StageTimeline& stage_timeline) const {
    return stage_timeline.stage_index == stage_index;
  }
};

std::unordered_map<std::uintptr_t, const NodeTimeline*>
build_array_id_to_timeline(const GpuTimeline& gpu_timeline) {
  std::unordered_map<std::uintptr_t, const NodeTimeline*> array_id_to_timeline;
  for (const NodeTimeline& node_timeline : gpu_timeline.nodes) {
    array_id_to_timeline.emplace(node_timeline.node->array_id, &node_timeline);
  }
  return array_id_to_timeline;
}

// The stage task a node ran in (per-stage pass), or nullptr.
const StageTimeline* find_stage_task(
    const GpuTimeline& gpu_timeline, const NodeTimeline* node_timeline) {
  if (node_timeline == nullptr || node_timeline->stage_position < 0) {
    return nullptr;
  }
  return &gpu_timeline.stages[node_timeline->stage_position];
}

// A per-stage pass did not time nodes, so their label names the stage
// instead of showing a per-node number it never measured.
std::string build_node_label(
    const GraphNode& captured_node,
    const NodeTimeline* node_timeline,
    const StageTimeline* stage_task) {
  std::ostringstream node_label;
  node_label << captured_node.label;
  if (!captured_node.op_name.empty()) {
    node_label << "\n" << captured_node.op_name;
  }
  node_label << "\n" << captured_node.shape_string << " "
             << captured_node.dtype_string << "  "
             << format_bytes(captured_node.output_bytes);
  if (stage_task != nullptr) {
    node_label << "\nstage: " << stage_task->stage_name;
  } else if (node_timeline != nullptr) {
    const LatencyBreakdown& breakdown = node_timeline->breakdown;
    node_label << "\n" << (breakdown.is_compute_measured ? "gpu " : "~")
               << format_microseconds(breakdown.compute_microseconds) << " · "
               << node_class_name(breakdown.classification);
  }
  return node_label.str();
}

std::string render_node_statement(
    const GraphNode& captured_node,
    const NodeTimeline* node_timeline,
    const StageTimeline* stage_task) {
  const LatencyBreakdown* timed_breakdown = stage_task != nullptr
      ? &stage_task->breakdown
      : (node_timeline != nullptr ? &node_timeline->breakdown : nullptr);
  const bool is_stalled_node = timed_breakdown != nullptr &&
      timed_breakdown->classification == NodeClass::wait_on_dependency;
  std::ostringstream node_statement;
  node_statement << "\"" << escape_dot_text(captured_node.label) << "\" [label=\""
                 << escape_dot_text(build_node_label(captured_node, node_timeline, stage_task))
                 << "\", shape=" << (captured_node.has_primitive ? "box" : "ellipse")
                 << ", style=filled, fillcolor=\""
                 << stream_fill_color(captured_node.stream_index) << "\""
                 << (is_stalled_node ? ", color=\"#e5484d\", penwidth=2.5" : "")
                 << "];\n";
  return node_statement.str();
}

// Spark's DAG view boxes each stage; here a per-stage pass gets one dashed
// cluster per stage. Input leaves and per-node passes stay unclustered.
void write_node_statements(
    const CapturedGraph& captured_graph,
    const GpuTimeline& gpu_timeline,
    std::ostringstream& plan_dot) {
  const auto array_id_to_timeline = build_array_id_to_timeline(gpu_timeline);
  std::map<int, std::string> stage_index_to_statements;
  for (const GraphNode& captured_node : captured_graph.nodes) {
    const auto timeline_entry = array_id_to_timeline.find(captured_node.array_id);
    const NodeTimeline* node_timeline =
        timeline_entry == array_id_to_timeline.end() ? nullptr : timeline_entry->second;
    const StageTimeline* stage_task = find_stage_task(gpu_timeline, node_timeline);
    const std::string node_statement =
        render_node_statement(captured_node, node_timeline, stage_task);
    if (stage_task == nullptr) {
      plan_dot << "  " << node_statement;
    } else {
      stage_index_to_statements[stage_task->stage_index] += "    " + node_statement;
    }
  }
  for (const auto& [stage_index, node_statements] : stage_index_to_statements) {
    const std::string& stage_name = std::find_if(
        gpu_timeline.stages.begin(), gpu_timeline.stages.end(),
        StageIndexMatcher{stage_index})->stage_name;
    plan_dot << "  subgraph \"cluster_stage_" << stage_index << "\" {\n    label=\""
             << escape_dot_text(stage_name) << "\"; style=dashed; color=\"#888888\";\n"
             << node_statements << "  }\n";
  }
}

} // namespace

std::string write_plan_dot(
    const CapturedGraph& captured_graph,
    const GpuTimeline& gpu_timeline,
    const std::string& report_title) {
  std::ostringstream plan_dot;
  plan_dot << "digraph \"" << escape_dot_text(report_title) << "\" {\n"
           << "  rankdir=TB;\n"
           << "  node [fontname=\"Helvetica\", fontsize=11];\n"
           << "  edge [fontname=\"Helvetica\", fontsize=9, color=\"#666666\"];\n";
  write_node_statements(captured_graph, gpu_timeline, plan_dot);
  for (const GraphEdge& captured_edge : captured_graph.edges) {
    plan_dot << "  \"" << escape_dot_text(captured_edge.from_label) << "\" -> \""
             << escape_dot_text(captured_edge.to_label) << "\" [label=\""
             << format_bytes(captured_edge.bytes) << "\"];\n";
  }
  plan_dot << "}\n";
  return plan_dot.str();
}

} // namespace mlx_perf
