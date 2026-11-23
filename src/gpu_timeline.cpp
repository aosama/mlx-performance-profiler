#include "gpu_timeline.h"

#include "progress_reporter.h"

#include "mlx/transforms.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>

namespace mlx_plan {
namespace {

using Clock = std::chrono::steady_clock;
constexpr int NO_POSITION = -1;

// Inputs of one node, as topological positions of their producers. Inputs
// with no producer in the plan (already-materialized leaves) are ready at
// pass start and appear in neither list.
struct NodeDependencies {
  std::vector<int> same_stream_input_positions;
  std::vector<int> cross_stream_input_positions;
  int previous_on_stream_position = NO_POSITION;
};

double microseconds_since(Clock::time_point pass_started_at) {
  return std::chrono::duration<double, std::micro>(
             Clock::now() - pass_started_at)
      .count();
}

const GraphNode& node_at(const CapturedGraph& captured_graph, int position) {
  return captured_graph.nodes[captured_graph.topological[position]];
}

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

// Must run before the first dispatch: evaluation detaches inputs.
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

// Tracks nodes issued but not yet observed complete, in topological order.
// Checking in that order and stamping each check individually guarantees a
// producer's observed completion never postdates its consumer's.
struct CompletionTracker {
  std::vector<int> pending_positions;
  std::vector<double> position_to_completed_microseconds;
  int completed_count = 0;
};

// One non-blocking sweep. The cost is O(pending) per call — small in
// practice because GPU work drains the pending set as fast as it is issued.
void sweep_completions(
    const CapturedGraph& captured_graph,
    Clock::time_point pass_started_at,
    CompletionTracker& tracker) {
  std::vector<int> still_pending_positions;
  for (const int position : tracker.pending_positions) {
    if (node_at(captured_graph, position).handle.is_available()) {
      tracker.position_to_completed_microseconds[position] =
          microseconds_since(pass_started_at);
      ++tracker.completed_count;
    } else {
      still_pending_positions.push_back(position);
    }
  }
  tracker.pending_positions = std::move(still_pending_positions);
}

// Issue instants per node (µs since pass start), filled during dispatch.
struct IssueInstants {
  std::vector<double> position_to_issue_started_microseconds;
  std::vector<double> position_to_issue_finished_microseconds;
};

// Dispatches every node with its own async_eval (one Event per node), then
// spins until all are observed complete. Sweeping between dispatches keeps
// early completions from being stamped late while the host is still issuing.
CompletionTracker dispatch_and_observe(
    const CapturedGraph& captured_graph,
    Clock::time_point pass_started_at,
    IssueInstants& issue_instants) {
  const int node_count = static_cast<int>(captured_graph.topological.size());
  CompletionTracker tracker;
  tracker.position_to_completed_microseconds.assign(node_count, 0.0);
  issue_instants.position_to_issue_started_microseconds.assign(node_count, 0.0);
  issue_instants.position_to_issue_finished_microseconds.assign(
      node_count, 0.0);

  ProgressReporter progress("gpu timeline nodes complete", node_count);
  for (int position = 0; position < node_count; ++position) {
    issue_instants.position_to_issue_started_microseconds[position] =
        microseconds_since(pass_started_at);
    mlx::core::async_eval({node_at(captured_graph, position).handle});
    issue_instants.position_to_issue_finished_microseconds[position] =
        microseconds_since(pass_started_at);
    tracker.pending_positions.push_back(position);
    sweep_completions(captured_graph, pass_started_at, tracker);
    progress.update(tracker.completed_count);
  }
  // Yield rather than sleep: a sleep's wake-up granularity (tens of µs or
  // more) would dominate the sub-millisecond kernels this pass resolves.
  while (!tracker.pending_positions.empty()) {
    std::this_thread::yield();
    sweep_completions(captured_graph, pass_started_at, tracker);
    progress.update(tracker.completed_count);
  }
  progress.finish(tracker.completed_count);
  return tracker;
}

double latest_completion(
    const std::vector<int>& producer_positions,
    const std::vector<double>& position_to_completed_microseconds) {
  double latest_microseconds = 0.0;
  for (const int producer_position : producer_positions) {
    latest_microseconds = std::max(
        latest_microseconds,
        position_to_completed_microseconds[producer_position]);
  }
  return latest_microseconds;
}

NodeTimestamps assemble_timestamps(
    int position,
    const NodeDependencies& dependencies,
    const IssueInstants& issue_instants,
    const std::vector<double>& position_to_completed_microseconds) {
  NodeTimestamps timestamps;
  timestamps.issue_started_microseconds =
      issue_instants.position_to_issue_started_microseconds[position];
  timestamps.issue_finished_microseconds =
      issue_instants.position_to_issue_finished_microseconds[position];
  timestamps.completed_microseconds =
      position_to_completed_microseconds[position];
  timestamps.stream_ready_microseconds = latest_completion(
      dependencies.same_stream_input_positions,
      position_to_completed_microseconds);
  if (dependencies.previous_on_stream_position != NO_POSITION) {
    timestamps.stream_ready_microseconds = std::max(
        timestamps.stream_ready_microseconds,
        position_to_completed_microseconds
            [dependencies.previous_on_stream_position]);
  }
  timestamps.cross_stream_inputs_ready_microseconds = latest_completion(
      dependencies.cross_stream_input_positions,
      position_to_completed_microseconds);
  return timestamps;
}

void accumulate_into_stream(
    const NodeTimeline& node_timeline, StreamTimeline& stream_timeline) {
  if (stream_timeline.node_count == 0) {
    stream_timeline.first_issue_microseconds =
        node_timeline.timestamps.issue_started_microseconds;
  }
  ++stream_timeline.node_count;
  stream_timeline.last_completion_microseconds = std::max(
      stream_timeline.last_completion_microseconds,
      node_timeline.timestamps.completed_microseconds);
  stream_timeline.busy_microseconds +=
      node_timeline.breakdown.compute_microseconds;
  stream_timeline.dependency_wait_microseconds +=
      node_timeline.breakdown.wait_on_dependency_microseconds;
}

// std::map keeps the streams ordered by index for stable reporting.
std::vector<StreamTimeline> build_stream_timelines(
    const std::vector<NodeTimeline>& node_timelines,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream) {
  std::map<int, StreamTimeline> stream_index_to_timeline;
  for (const NodeTimeline& node_timeline : node_timelines) {
    const int stream_index = node_timeline.node->stream_index;
    StreamTimeline& stream_timeline = stream_index_to_timeline[stream_index];
    stream_timeline.stream_index = stream_index;
    stream_timeline.is_gpu = stream_index_to_stream.at(stream_index).device ==
        mlx::core::Device::gpu;
    accumulate_into_stream(node_timeline, stream_timeline);
  }
  std::vector<StreamTimeline> stream_timelines;
  for (auto& [stream_index, stream_timeline] : stream_index_to_timeline) {
    stream_timeline.wall_microseconds =
        stream_timeline.last_completion_microseconds -
        stream_timeline.first_issue_microseconds;
    stream_timelines.push_back(stream_timeline);
  }
  return stream_timelines;
}

// Checked before dispatch, so a caller error fails fast and leaves the graph
// unevaluated instead of throwing after it has been consumed.
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

} // namespace

GpuTimeline capture_gpu_timeline(
    const CapturedGraph& captured_graph,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream) {
  require_every_stream_known(captured_graph, stream_index_to_stream);
  const std::vector<NodeDependencies> position_to_dependencies =
      snapshot_dependencies(captured_graph);
  IssueInstants issue_instants;
  const Clock::time_point pass_started_at = Clock::now();
  const CompletionTracker tracker =
      dispatch_and_observe(captured_graph, pass_started_at, issue_instants);

  GpuTimeline gpu_timeline;
  const int node_count = static_cast<int>(captured_graph.topological.size());
  for (int position = 0; position < node_count; ++position) {
    NodeTimeline node_timeline;
    node_timeline.node = &node_at(captured_graph, position);
    node_timeline.timestamps = assemble_timestamps(
        position,
        position_to_dependencies[position],
        issue_instants,
        tracker.position_to_completed_microseconds);
    node_timeline.breakdown = decompose_node_latency(node_timeline.timestamps);
    node_timeline.has_cross_stream_input =
        !position_to_dependencies[position].cross_stream_input_positions.empty();
    gpu_timeline.cross_stream_stall_microseconds +=
        node_timeline.breakdown.wait_on_dependency_microseconds;
    gpu_timeline.timeline_wall_microseconds = std::max(
        gpu_timeline.timeline_wall_microseconds,
        node_timeline.timestamps.completed_microseconds);
    gpu_timeline.nodes.push_back(node_timeline);
  }
  gpu_timeline.streams =
      build_stream_timelines(gpu_timeline.nodes, stream_index_to_stream);
  return gpu_timeline;
}

} // namespace mlx_plan
