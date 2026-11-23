#include "dot_writer.h"

#include <sstream>
#include <unordered_map>

namespace mlx_plan {
namespace {

// Spark colors stages; here one hue per stream index. Chosen to stay legible
// with black text at small SVG sizes.
const char* stream_fill_color(int stream_index) {
  static const char* fill_colors[] = {
      "#cfe2ff", // blue — default stream
      "#d8f0d8", // green
      "#fde8cf", // orange
      "#e8d8f0", // purple
      "#cfe8e8", // teal
  };
  return fill_colors[stream_index % 5];
}

std::string format_microseconds(double duration_microseconds) {
  if (duration_microseconds >= 1000.0) {
    std::ostringstream duration_text;
    duration_text << duration_microseconds / 1000.0 << " ms";
    return duration_text.str();
  }
  std::ostringstream duration_text;
  duration_text << duration_microseconds << " µs";
  return duration_text.str();
}

std::string format_bytes(size_t byte_count) {
  const double kib = 1024.0;
  if (byte_count >= 1024 * 1024) {
    return std::to_string(byte_count / (1024 * 1024)) + " MiB";
  }
  if (static_cast<double>(byte_count) >= kib) {
    std::ostringstream bytes_text;
    bytes_text << byte_count / kib << " KiB";
    return bytes_text.str();
  }
  return std::to_string(byte_count) + " B";
}

std::string escape_dot_text(const std::string& raw_text) {
  std::string escaped_text;
  for (const char letter : raw_text) {
    if (letter == '"' || letter == '\\') {
      escaped_text += '\\';
    }
    escaped_text += letter;
  }
  return escaped_text;
}

std::unordered_map<std::uintptr_t, const NodeTelemetry&>
build_node_id_to_telemetry(const RunTelemetry& run_telemetry) {
  std::unordered_map<std::uintptr_t, const NodeTelemetry&> node_id_to_telemetry;
  for (const NodeTelemetry& node_telemetry : run_telemetry.attributed_nodes) {
    node_id_to_telemetry.emplace(
        node_telemetry.node->array_id, node_telemetry);
  }
  return node_id_to_telemetry;
}

} // namespace

std::string write_plan_dot(
    const CapturedGraph& captured_graph,
    const RunTelemetry& run_telemetry,
    const std::string& report_title) {
  const std::unordered_map<std::uintptr_t, const NodeTelemetry&>
      node_id_to_telemetry = build_node_id_to_telemetry(run_telemetry);

  std::ostringstream plan_dot;
  plan_dot << "digraph \"" << escape_dot_text(report_title) << "\" {\n";
  plan_dot << "  rankdir=TB;\n";
  plan_dot << "  node [fontname=\"Helvetica\", fontsize=11];\n";
  plan_dot << "  edge [fontname=\"Helvetica\", fontsize=9, color=\"#666666\"];\n";

  for (const GraphNode& captured_node : captured_graph.nodes) {
    std::ostringstream node_label;
    node_label << captured_node.label;
    if (!captured_node.op_name.empty()) {
      node_label << "\\n" << captured_node.op_name;
    }
    if (captured_node.has_primitive) {
      const auto telemetry_entry = node_id_to_telemetry.find(captured_node.array_id);
      if (telemetry_entry != node_id_to_telemetry.end()) {
        node_label << "\\n"
                   << captured_node.shape_string << " "
                   << captured_node.dtype_string << "\\n"
                   << format_microseconds(
                          telemetry_entry->second.duration_microseconds)
                   << "  "
                   << format_bytes(captured_node.output_bytes);
      }
    } else {
      node_label << "\\n" << captured_node.shape_string << " "
                 << captured_node.dtype_string << "\\n"
                 << format_bytes(captured_node.output_bytes);
    }
    plan_dot << "  \"" << captured_node.label << "\" [label=\""
             << escape_dot_text(node_label.str())
             << "\", shape=" << (captured_node.has_primitive ? "box" : "ellipse")
             << ", style=filled, fillcolor=\""
             << stream_fill_color(captured_node.stream_index) << "\"];\n";
  }

  for (const GraphEdge& captured_edge : captured_graph.edges) {
    plan_dot << "  \"" << captured_edge.from_label << "\" -> \""
             << captured_edge.to_label << "\" [label=\""
             << format_bytes(captured_edge.bytes) << "\"];\n";
  }

  plan_dot << "}\n";
  return plan_dot.str();
}

} // namespace mlx_plan
