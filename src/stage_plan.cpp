#include "stage_plan.h"

#include <stdexcept>
#include <string_view>

namespace mlx_perf {
namespace {

constexpr int UNASSIGNED_STAGE = -1;

void require_topological_inputs(
    const std::vector<std::vector<int>>& position_to_input_positions) {
  const int node_count = static_cast<int>(position_to_input_positions.size());
  for (int position = 0; position < node_count; ++position) {
    for (const int input_position : position_to_input_positions[position]) {
      if (input_position < 0 || input_position >= position) {
        throw std::invalid_argument(
            "plan_stages: input " + std::to_string(input_position) +
            " of node " + std::to_string(position) +
            " does not precede it in topological order");
      }
    }
  }
}

// Claims every unassigned ancestor of the cut (and the cut itself) for
// `stage_index`. An already-assigned node stops the walk: its ancestors were
// claimed by the same earlier stage or one before it. Iterative so a deep
// chain (thousands of nodes) cannot overflow the call stack.
int claim_ancestors(
    const std::vector<std::vector<int>>& position_to_input_positions,
    const std::vector<int>& boundary_positions,
    int stage_index,
    std::vector<int>& position_to_stage) {
  const int node_count = static_cast<int>(position_to_stage.size());
  std::vector<int> positions_to_visit;
  for (const int boundary_position : boundary_positions) {
    if (boundary_position < 0 || boundary_position >= node_count) {
      throw std::invalid_argument(
          "plan_stages: boundary position " + std::to_string(boundary_position) +
          " is outside the graph");
    }
    positions_to_visit.push_back(boundary_position);
  }
  int claimed_count = 0;
  while (!positions_to_visit.empty()) {
    const int position = positions_to_visit.back();
    positions_to_visit.pop_back();
    if (position_to_stage[position] != UNASSIGNED_STAGE) {
      continue;
    }
    position_to_stage[position] = stage_index;
    ++claimed_count;
    for (const int input_position : position_to_input_positions[position]) {
      positions_to_visit.push_back(input_position);
    }
  }
  return claimed_count;
}

std::vector<std::vector<int>> find_stage_sinks(
    const std::vector<std::vector<int>>& position_to_input_positions,
    const std::vector<int>& position_to_stage,
    int stage_count) {
  const int node_count = static_cast<int>(position_to_stage.size());
  std::vector<bool> is_consumed_within_stage(node_count, false);
  for (int position = 0; position < node_count; ++position) {
    for (const int input_position : position_to_input_positions[position]) {
      if (position_to_stage[input_position] == position_to_stage[position]) {
        is_consumed_within_stage[input_position] = true;
      }
    }
  }
  std::vector<std::vector<int>> stage_to_sink_positions(stage_count);
  for (int position = 0; position < node_count; ++position) {
    if (!is_consumed_within_stage[position]) {
      stage_to_sink_positions[position_to_stage[position]].push_back(position);
    }
  }
  return stage_to_sink_positions;
}

} // namespace

TimelineGranularity choose_granularity(
    GranularityChoice granularity_choice, int stage_cut_count, int primitive_count) {
  switch (granularity_choice) {
    case GranularityChoice::per_node: return TimelineGranularity::per_node;
    case GranularityChoice::per_stage: return TimelineGranularity::per_stage;
    case GranularityChoice::automatic: break;
  }
  const bool is_large_enough =
      primitive_count >= AUTOMATIC_STAGE_MINIMUM_PRIMITIVE_COUNT;
  return stage_cut_count > 0 && is_large_enough ? TimelineGranularity::per_stage
                                                : TimelineGranularity::per_node;
}

std::string_view granularity_name(TimelineGranularity granularity) {
  return granularity == TimelineGranularity::per_stage ? "per-stage" : "per-node";
}

StagePlan plan_stages(
    const std::vector<std::vector<int>>& position_to_input_positions,
    const std::vector<StageCut>& stage_cuts,
    const std::string& final_stage_name) {
  require_topological_inputs(position_to_input_positions);
  const int node_count = static_cast<int>(position_to_input_positions.size());
  StagePlan stage_plan;
  stage_plan.position_to_stage.assign(node_count, UNASSIGNED_STAGE);
  for (const StageCut& stage_cut : stage_cuts) {
    const int stage_index = static_cast<int>(stage_plan.stage_names.size());
    // A cut that claims nothing new (its boundary was already inside an
    // earlier stage) would be an empty dispatch; it is dropped, not timed.
    if (claim_ancestors(position_to_input_positions, stage_cut.boundary_positions,
                        stage_index, stage_plan.position_to_stage) > 0) {
      stage_plan.stage_names.push_back(stage_cut.stage_name);
    }
  }
  const int final_stage_index = static_cast<int>(stage_plan.stage_names.size());
  bool has_unclaimed_nodes = false;
  for (int& stage_index : stage_plan.position_to_stage) {
    if (stage_index == UNASSIGNED_STAGE) {
      stage_index = final_stage_index;
      has_unclaimed_nodes = true;
    }
  }
  if (has_unclaimed_nodes) {
    stage_plan.stage_names.push_back(final_stage_name);
  }
  stage_plan.stage_to_sink_positions = find_stage_sinks(
      position_to_input_positions, stage_plan.position_to_stage,
      static_cast<int>(stage_plan.stage_names.size()));
  return stage_plan;
}

} // namespace mlx_perf
