#include "dot_writer.h"

#include "text_format.h"

#include <sstream>
#include <unordered_map>

namespace mlx_plan {
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

std::unordered_map<std::uintptr_t, const NodeTimeline*>
build_array_id_to_timeline(const GpuTimeline& gpu_timeline) {
  std::unordered_map<std::uintptr_t, const NodeTimeline*> array_id_to_timeline;
  for (const NodeTimeline& node_timeline : gpu_timeline.nodes) {
    array_id_to_timeline.emplace(node_timeline.node->array_id, &node_timeline);
  }
  return array_id_to_timeline;
}

std::string build_node_label(
    const GraphNode& captured_node, const NodeTimeline* node_timeline) {
  std::ostringstream node_label;
  node_label << captured_node.label;
  if (!captured_node.op_name.empty()) {
    node_label << "\n" << captured_node.op_name;
  }
  node_label << "\n" << captured_node.shape_string << " "
             << captured_node.dtype_string << "  "
             << format_bytes(captured_node.output_bytes);
  if (node_timeline != nullptr) {
    const LatencyBreakdown& breakdown = node_timeline->breakdown;
    node_label << "\n" << (breakdown.is_compute_measured ? "gpu " : "~")
               << format_microseconds(breakdown.compute_microseconds) << " · "
               << node_class_name(breakdown.classification);
  }
  return node_label.str();
}

} // namespace

std::string write_plan_dot(
    const CapturedGraph& captured_graph,
    const GpuTimeline& gpu_timeline,
    const std::string& report_title) {
  const auto array_id_to_timeline = build_array_id_to_timeline(gpu_timeline);
  std::ostringstream plan_dot;
  plan_dot << "digraph \"" << escape_dot_text(report_title) << "\" {\n"
           << "  rankdir=TB;\n"
           << "  node [fontname=\"Helvetica\", fontsize=11];\n"
           << "  edge [fontname=\"Helvetica\", fontsize=9, color=\"#666666\"];\n";
  for (const GraphNode& captured_node : captured_graph.nodes) {
    const auto timeline_entry = array_id_to_timeline.find(captured_node.array_id);
    const NodeTimeline* node_timeline =
        timeline_entry == array_id_to_timeline.end() ? nullptr
                                                     : timeline_entry->second;
    const bool is_stalled_node = node_timeline != nullptr &&
        node_timeline->breakdown.classification == NodeClass::wait_on_dependency;
    plan_dot << "  \"" << escape_dot_text(captured_node.label) << "\" [label=\""
             << escape_dot_text(build_node_label(captured_node, node_timeline))
             << "\", shape=" << (captured_node.has_primitive ? "box" : "ellipse")
             << ", style=filled, fillcolor=\""
             << stream_fill_color(captured_node.stream_index) << "\""
             << (is_stalled_node ? ", color=\"#e5484d\", penwidth=2.5" : "")
             << "];\n";
  }
  for (const GraphEdge& captured_edge : captured_graph.edges) {
    plan_dot << "  \"" << escape_dot_text(captured_edge.from_label) << "\" -> \""
             << escape_dot_text(captured_edge.to_label) << "\" [label=\""
             << format_bytes(captured_edge.bytes) << "\"];\n";
  }
  plan_dot << "}\n";
  return plan_dot.str();
}

} // namespace mlx_plan
