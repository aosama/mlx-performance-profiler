// Unit tests for stage_plan: pure grouping rules over topological positions,
// no GPU needed.

#include "stage_plan.h"
#include "test_harness.h"

#include <stdexcept>
#include <vector>

namespace {

using mlx_plan::GranularityChoice;
using mlx_plan::StageCut;
using mlx_plan::StagePlan;
using mlx_plan::TimelineGranularity;

// A two-layer chain: 0 → 1 → 2 (layer 0 ends at 2) → 3 → 4 (layer 1 ends at
// 4) → 5 (head). Node 1 also feeds node 3, a skip connection like a residual.
std::vector<std::vector<int>> make_two_layer_chain() {
  return {{}, {0}, {1}, {2, 1}, {3}, {4}};
}

// --- positive ---

void should_assign_each_node_to_the_first_cut_that_needs_it() {
  const StagePlan stage_plan = mlx_plan::plan_stages(
      make_two_layer_chain(), {{"layer 0", {2}}, {"layer 1", {4}}}, "head");
  EXPECT_TRUE(stage_plan.stage_names ==
              (std::vector<std::string>{"layer 0", "layer 1", "head"}));
  EXPECT_TRUE(stage_plan.position_to_stage == (std::vector<int>{0, 0, 0, 1, 1, 2}));
}

void should_target_only_the_sinks_of_each_stage() {
  const StagePlan stage_plan = mlx_plan::plan_stages(
      make_two_layer_chain(), {{"layer 0", {2}}, {"layer 1", {4}}}, "head");
  EXPECT_TRUE(stage_plan.stage_to_sink_positions ==
              (std::vector<std::vector<int>>{{2}, {4}, {5}}));
}

// A stage with two independent outputs (hidden state and a cache) has two
// sinks; one async_eval must target both.
void should_keep_every_independent_output_of_a_stage_as_a_sink() {
  const std::vector<std::vector<int>> forked_layer = {{}, {0}, {0}, {1, 2}};
  const StagePlan stage_plan =
      mlx_plan::plan_stages(forked_layer, {{"layer", {1, 2}}}, "rest");
  EXPECT_TRUE(stage_plan.stage_to_sink_positions.front() == (std::vector<int>{1, 2}));
  EXPECT_TRUE(stage_plan.position_to_stage == (std::vector<int>{0, 0, 0, 1}));
}

void should_choose_stages_automatically_only_for_large_staged_graphs() {
  const int large_count = mlx_plan::AUTOMATIC_STAGE_MINIMUM_PRIMITIVE_COUNT;
  EXPECT_TRUE(mlx_plan::choose_granularity(GranularityChoice::automatic, 24, large_count) ==
              TimelineGranularity::per_stage);
  EXPECT_TRUE(mlx_plan::choose_granularity(GranularityChoice::per_node, 24, large_count) ==
              TimelineGranularity::per_node);
  EXPECT_TRUE(mlx_plan::choose_granularity(GranularityChoice::per_stage, 0, 3) ==
              TimelineGranularity::per_stage);
}

// --- edge / boundary ---

void should_keep_per_node_just_below_the_automatic_threshold_and_without_cuts() {
  const int threshold = mlx_plan::AUTOMATIC_STAGE_MINIMUM_PRIMITIVE_COUNT;
  EXPECT_TRUE(mlx_plan::choose_granularity(GranularityChoice::automatic, 24, threshold - 1) ==
              TimelineGranularity::per_node);
  EXPECT_TRUE(mlx_plan::choose_granularity(GranularityChoice::automatic, 0, 100000) ==
              TimelineGranularity::per_node);
}

void should_make_the_whole_graph_one_stage_when_there_are_no_cuts() {
  const StagePlan stage_plan = mlx_plan::plan_stages(make_two_layer_chain(), {}, "whole job");
  EXPECT_TRUE(stage_plan.stage_names == (std::vector<std::string>{"whole job"}));
  EXPECT_TRUE(stage_plan.stage_to_sink_positions == (std::vector<std::vector<int>>{{5}}));
}

// A cut already inside an earlier stage, or with no boundary at all, would
// be an empty dispatch; it must vanish rather than be timed as zero.
void should_drop_cuts_that_claim_nothing_new() {
  const StagePlan stage_plan = mlx_plan::plan_stages(
      make_two_layer_chain(),
      {{"layer 0", {2}}, {"inside layer 0", {1}}, {"empty", {}}, {"layer 1", {4}}}, "head");
  EXPECT_TRUE(stage_plan.stage_names ==
              (std::vector<std::string>{"layer 0", "layer 1", "head"}));
  EXPECT_TRUE(stage_plan.position_to_stage == (std::vector<int>{0, 0, 0, 1, 1, 2}));
}

void should_omit_the_final_stage_when_cuts_cover_everything() {
  const StagePlan stage_plan =
      mlx_plan::plan_stages(make_two_layer_chain(), {{"all", {5}}}, "head");
  EXPECT_TRUE(stage_plan.stage_names == (std::vector<std::string>{"all"}));
}

void should_plan_an_empty_graph_as_no_stages() {
  const StagePlan stage_plan = mlx_plan::plan_stages({}, {{"layer 0", {}}}, "head");
  EXPECT_TRUE(stage_plan.stage_names.empty());
  EXPECT_TRUE(stage_plan.position_to_stage.empty());
}

// --- negative ---

void should_reject_a_boundary_outside_the_graph() {
  bool has_thrown = false;
  try {
    mlx_plan::plan_stages(make_two_layer_chain(), {{"bad", {6}}}, "head");
  } catch (const std::invalid_argument&) {
    has_thrown = true;
  }
  EXPECT_TRUE(has_thrown);
}

void should_reject_inputs_that_are_not_in_topological_order() {
  bool has_thrown = false;
  try {
    mlx_plan::plan_stages({{1}, {}}, {}, "head");
  } catch (const std::invalid_argument&) {
    has_thrown = true;
  }
  EXPECT_TRUE(has_thrown);
}

} // namespace

int main() {
  return test_harness::run_all({
      {"should_assign_each_node_to_the_first_cut_that_needs_it",
       should_assign_each_node_to_the_first_cut_that_needs_it},
      {"should_target_only_the_sinks_of_each_stage", should_target_only_the_sinks_of_each_stage},
      {"should_keep_every_independent_output_of_a_stage_as_a_sink",
       should_keep_every_independent_output_of_a_stage_as_a_sink},
      {"should_choose_stages_automatically_only_for_large_staged_graphs",
       should_choose_stages_automatically_only_for_large_staged_graphs},
      {"should_keep_per_node_just_below_the_automatic_threshold_and_without_cuts",
       should_keep_per_node_just_below_the_automatic_threshold_and_without_cuts},
      {"should_make_the_whole_graph_one_stage_when_there_are_no_cuts",
       should_make_the_whole_graph_one_stage_when_there_are_no_cuts},
      {"should_drop_cuts_that_claim_nothing_new", should_drop_cuts_that_claim_nothing_new},
      {"should_omit_the_final_stage_when_cuts_cover_everything",
       should_omit_the_final_stage_when_cuts_cover_everything},
      {"should_plan_an_empty_graph_as_no_stages", should_plan_an_empty_graph_as_no_stages},
      {"should_reject_a_boundary_outside_the_graph", should_reject_a_boundary_outside_the_graph},
      {"should_reject_inputs_that_are_not_in_topological_order",
       should_reject_inputs_that_are_not_in_topological_order},
  });
}
