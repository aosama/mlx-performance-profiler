#include "gpu_timeline.h"

#include "metal_command_buffer_probe.h"
#include "progress_reporter.h"
#include "timed_units.h"
#include "timeline_dependencies.h"

#include "mlx/transforms.h"

#include <algorithm>
#include <chrono>
#include <thread>

namespace mlx_perf {
namespace {

using Clock = TimelineClock;

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
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream,
    Clock::time_point pass_started_at,
    IssueInstants& issue_instants,
    MetalCommandBufferProbe& command_buffer_probe) {
  const int node_count = static_cast<int>(captured_graph.topological.size());
  CompletionTracker tracker;
  tracker.position_to_completed_microseconds.assign(node_count, 0.0);
  issue_instants.position_to_issue_started_microseconds.assign(node_count, 0.0);
  issue_instants.position_to_issue_finished_microseconds.assign(
      node_count, 0.0);

  ProgressReporter progress("gpu timeline nodes complete", node_count);
  for (int position = 0; position < node_count; ++position) {
    const GraphNode& graph_node = node_at(captured_graph, position);
    // Taken before issue_started so the probe's own cost (a pointer read
    // and a retain) is not charged to the node as host issue time.
    command_buffer_probe.hold_buffer_for(
        position, stream_index_to_stream.at(graph_node.stream_index));
    issue_instants.position_to_issue_started_microseconds[position] =
        microseconds_since(pass_started_at);
    mlx::core::async_eval({graph_node.handle});
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

NodeTimestamps assemble_timestamps(
    int position,
    const NodeDependencies& dependencies,
    const IssueInstants& issue_instants,
    const std::vector<double>& position_to_completed_microseconds,
    const std::vector<double>& position_to_ready_microseconds,
    const GpuExecutionWindow& gpu_window) {
  NodeTimestamps timestamps;
  timestamps.has_gpu_window = gpu_window.is_measured;
  timestamps.gpu_started_microseconds = gpu_window.started_microseconds;
  timestamps.gpu_finished_microseconds = gpu_window.finished_microseconds;
  timestamps.issue_started_microseconds =
      issue_instants.position_to_issue_started_microseconds[position];
  timestamps.issue_finished_microseconds =
      issue_instants.position_to_issue_finished_microseconds[position];
  timestamps.completed_microseconds =
      position_to_completed_microseconds[position];
  timestamps.stream_ready_microseconds = latest_ready(
      dependencies.same_stream_input_positions,
      position_to_ready_microseconds);
  if (dependencies.previous_on_stream_position != NO_POSITION) {
    timestamps.stream_ready_microseconds = std::max(
        timestamps.stream_ready_microseconds,
        position_to_ready_microseconds
            [dependencies.previous_on_stream_position]);
  }
  timestamps.cross_stream_inputs_ready_microseconds = latest_ready(
      dependencies.cross_stream_input_positions,
      position_to_ready_microseconds);
  return timestamps;
}

} // namespace

GpuTimeline capture_gpu_timeline(
    const CapturedGraph& captured_graph,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream) {
  require_every_stream_known(captured_graph, stream_index_to_stream);
  const std::vector<NodeDependencies> position_to_dependencies =
      snapshot_dependencies(captured_graph);
  IssueInstants issue_instants;
  const int node_count = static_cast<int>(captured_graph.topological.size());
  const Clock::time_point pass_started_at = Clock::now();
  MetalCommandBufferProbe command_buffer_probe(node_count, pass_started_at);
  const CompletionTracker tracker = dispatch_and_observe(
      captured_graph, stream_index_to_stream, pass_started_at, issue_instants,
      command_buffer_probe);
  const std::vector<GpuExecutionWindow> position_to_gpu_window =
      command_buffer_probe.read_windows();
  const std::vector<double> position_to_ready_microseconds =
      build_ready_microseconds(
          tracker.position_to_completed_microseconds, position_to_gpu_window);

  GpuTimeline gpu_timeline;
  for (int position = 0; position < node_count; ++position) {
    NodeTimeline node_timeline;
    node_timeline.node = &node_at(captured_graph, position);
    node_timeline.timestamps = assemble_timestamps(
        position,
        position_to_dependencies[position],
        issue_instants,
        tracker.position_to_completed_microseconds,
        position_to_ready_microseconds,
        position_to_gpu_window[position]);
    node_timeline.breakdown = decompose_node_latency(node_timeline.timestamps);
    record_structure(position_to_dependencies[position], node_timeline);
    gpu_timeline.cross_stream_stall_microseconds +=
        node_timeline.breakdown.wait_on_dependency_microseconds;
    if (node_timeline.breakdown.is_compute_measured) {
      ++gpu_timeline.gpu_measured_node_count;
      gpu_timeline.measured_gpu_compute_microseconds +=
          node_timeline.breakdown.compute_microseconds;
    }
    gpu_timeline.timeline_wall_microseconds = std::max(
        gpu_timeline.timeline_wall_microseconds,
        node_timeline.timestamps.completed_microseconds);
    gpu_timeline.nodes.push_back(node_timeline);
  }
  gpu_timeline.observed_command_buffer_count =
      command_buffer_probe.count_completed_command_buffers();
  gpu_timeline.streams = summarize_streams(
      collect_timed_units(gpu_timeline), stream_index_to_stream);
  return gpu_timeline;
}

} // namespace mlx_perf
