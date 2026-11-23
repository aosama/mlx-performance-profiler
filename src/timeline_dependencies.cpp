#include "timeline_dependencies.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace mlx_plan {

// Sibling outputs of a multi-output primitive complete with that primitive,
// so every sibling id resolves to the primitive's own position.
std::unordered_map<std::uintptr_t, int> build_array_id_to_position(
    const CapturedGraph& captured_graph) {
  std::unordered_map<std::uintptr_t, int> array_id_to_position;
  const int node_count = static_cast<int>(captured_graph.topological.size());
  for (int position = 0; position < node_count; ++position) {
    const GraphNode& graph_node = node_at(captured_graph, position);
    array_id_to_position.emplace(graph_node.array_id, position);
    for (const auto& sibling : graph_node.handle.siblings()) {
      array_id_to_position.emplace(sibling.id(), position);
    }
  }
  return array_id_to_position;
}

double microseconds_since(TimelineClock::time_point pass_started_at) {
  return std::chrono::duration<double, std::micro>(
             TimelineClock::now() - pass_started_at)
      .count();
}

const GraphNode& node_at(const CapturedGraph& captured_graph, int position) {
  return captured_graph.nodes[captured_graph.topological[position]];
}

std::vector<NodeDependencies> snapshot_dependencies(
    const CapturedGraph& captured_graph) {
  const auto array_id_to_position = build_array_id_to_position(captured_graph);
  const int node_count = static_cast<int>(captured_graph.topological.size());
  std::vector<NodeDependencies> position_to_dependencies(node_count);
  std::unordered_map<int, int> stream_index_to_last_position;
  for (int position = 0; position < node_count; ++position) {
    const GraphNode& graph_node = node_at(captured_graph, position);
    NodeDependencies& dependencies = position_to_dependencies[position];
    for (const auto& input : graph_node.handle.inputs()) {
      const auto producer = array_id_to_position.find(input.id());
      if (producer == array_id_to_position.end()) {
        continue;
      }
      const bool is_same_stream =
          node_at(captured_graph, producer->second).stream_index ==
          graph_node.stream_index;
      (is_same_stream ? dependencies.same_stream_input_positions
                      : dependencies.cross_stream_input_positions)
          .push_back(producer->second);
    }
    const auto last_on_stream =
        stream_index_to_last_position.find(graph_node.stream_index);
    if (last_on_stream != stream_index_to_last_position.end()) {
      dependencies.previous_on_stream_position = last_on_stream->second;
    }
    stream_index_to_last_position[graph_node.stream_index] = position;
  }
  return position_to_dependencies;
}

std::vector<std::vector<int>> merge_input_positions(
    const std::vector<NodeDependencies>& position_to_dependencies) {
  std::vector<std::vector<int>> position_to_input_positions;
  position_to_input_positions.reserve(position_to_dependencies.size());
  for (const NodeDependencies& dependencies : position_to_dependencies) {
    std::vector<int> input_positions = dependencies.same_stream_input_positions;
    input_positions.insert(input_positions.end(),
                           dependencies.cross_stream_input_positions.begin(),
                           dependencies.cross_stream_input_positions.end());
    position_to_input_positions.push_back(std::move(input_positions));
  }
  return position_to_input_positions;
}

void require_every_stream_known(
    const CapturedGraph& captured_graph,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream) {
  for (const std::size_t node_index : captured_graph.topological) {
    const int stream_index = captured_graph.nodes[node_index].stream_index;
    if (stream_index_to_stream.find(stream_index) ==
        stream_index_to_stream.end()) {
      throw std::out_of_range(
          "capture_gpu_timeline: no Stream supplied for stream index " +
          std::to_string(stream_index));
    }
  }
}

std::vector<double> build_ready_microseconds(
    const std::vector<double>& unit_to_completed_microseconds,
    const std::vector<GpuExecutionWindow>& unit_to_gpu_window) {
  std::vector<double> unit_to_ready_microseconds = unit_to_completed_microseconds;
  for (std::size_t unit = 0; unit < unit_to_gpu_window.size(); ++unit) {
    if (unit_to_gpu_window[unit].is_measured) {
      unit_to_ready_microseconds[unit] =
          unit_to_gpu_window[unit].finished_microseconds;
    }
  }
  return unit_to_ready_microseconds;
}

double latest_ready(
    const std::vector<int>& producer_units,
    const std::vector<double>& unit_to_ready_microseconds) {
  double latest_microseconds = 0.0;
  for (const int producer_unit : producer_units) {
    latest_microseconds =
        std::max(latest_microseconds, unit_to_ready_microseconds[producer_unit]);
  }
  return latest_microseconds;
}

} // namespace mlx_plan
