#pragma once

// Timeline dependencies: one concern only — the dependency facts both
// timeline passes (per-node and per-stage) need before dispatch, and the
// producer-readiness arithmetic they share after it. Snapshotted up front
// because MLX detaches a node's inputs once it is evaluated.

#include "gpu_timeline.h"
#include "graph_model.h"
#include "metal_command_buffer_probe.h"

#include "mlx/stream.h"

#include <chrono>
#include <unordered_map>
#include <vector>

namespace mlx_perf {

using TimelineClock = std::chrono::steady_clock;
inline constexpr int NO_POSITION = -1;

// Inputs of one node, as topological positions of their producers. Inputs
// with no producer in the plan (already-materialized leaves) are ready at
// pass start and appear in neither list.
struct NodeDependencies {
  std::vector<int> same_stream_input_positions;
  std::vector<int> cross_stream_input_positions;
  int previous_on_stream_position = NO_POSITION;
};

double microseconds_since(TimelineClock::time_point pass_started_at);

const GraphNode& node_at(const CapturedGraph& captured_graph, int position);

// Every primitive's array id — and each sibling output's id, which resolves
// to the primitive that produces it — mapped to its topological position.
std::unordered_map<std::uintptr_t, int> build_array_id_to_position(
    const CapturedGraph& captured_graph);

// Must run before the first dispatch: evaluation detaches inputs.
std::vector<NodeDependencies> snapshot_dependencies(
    const CapturedGraph& captured_graph);

// Same- and cross-stream inputs together: what stage planning walks.
std::vector<std::vector<int>> merge_input_positions(
    const std::vector<NodeDependencies>& position_to_dependencies);

// Copies one node's dependency facts onto its timeline entry: producers on
// other streams (the Gantt's arrows), all producers in position order, and
// the previous node on its stream. Shared by both timeline passes.
void record_structure(const NodeDependencies& dependencies, NodeTimeline& node_timeline);

// Checked before dispatch, so a caller error fails fast and leaves the graph
// unevaluated instead of throwing after it has been consumed.
void require_every_stream_known(
    const CapturedGraph& captured_graph,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream);

// When each timed unit's data was really ready: its GPU end when the probe
// measured one (exact), else the host-observed completion (trails reality).
std::vector<double> build_ready_microseconds(
    const std::vector<double>& unit_to_completed_microseconds,
    const std::vector<GpuExecutionWindow>& unit_to_gpu_window);

// Latest ready instant among `producer_units`; 0 when there are none.
double latest_ready(
    const std::vector<int>& producer_units,
    const std::vector<double>& unit_to_ready_microseconds);

} // namespace mlx_perf
