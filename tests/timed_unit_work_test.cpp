// Unit tests for timed_unit_work: a node's work joined to its launches, a
// stage task's work summed over its members, and each judged against its own
// GPU window. Synthetic runs, no GPU.

#include "report_fixture.h"
#include "test_harness.h"
#include "timed_unit_work.h"

namespace {

using namespace mlx_perf;

const MachineCeilings ROUND_CEILINGS{true, 1e9, 1e12, "bfloat16"};

KernelDispatch one_launch() {
  KernelDispatch kernel_dispatch;
  kernel_dispatch.grid = {64, 1, 1};
  return kernel_dispatch;
}

// Both nodes carry 40,000 bytes of work; the producer's 100 µs window and
// the join's 7 µs window come from the fixture.
template <typename MeasuredRun>
void give_both_nodes_work(MeasuredRun& synthetic_run) {
  for (GraphNode& graph_node : synthetic_run.timeline_graph.nodes) {
    graph_node.work.bytes_read = 40'000;
  }
}

// --- positive ---

void should_judge_a_node_against_its_own_gpu_window() {
  report_fixture::SyntheticRun synthetic_run;
  report_fixture::populate_two_stream_run(synthetic_run);
  give_both_nodes_work(synthetic_run);
  const WorkJudgement judgement =
      judge_node(synthetic_run.gpu_timeline, 0, synthetic_run.serial_telemetry, ROUND_CEILINGS);
  EXPECT_NEAR(judgement.data_movement_share, 0.4, 1e-9);
  EXPECT_TRUE(judgement.limited_by == WorkLimit::neither);
}

void should_sum_a_stage_tasks_members() {
  report_fixture::SyntheticRun synthetic_run;
  report_fixture::populate_two_stage_run(synthetic_run);
  give_both_nodes_work(synthetic_run);
  StageTimeline both_layers = synthetic_run.gpu_timeline.stages[0];
  both_layers.node_positions = {0, 1};
  EXPECT_TRUE(work_of_stage(synthetic_run.gpu_timeline, both_layers, synthetic_run.serial_telemetry)
                  .bytes_read == 80'000);
  const WorkJudgement judgement =
      judge_stage(synthetic_run.gpu_timeline, both_layers, synthetic_run.serial_telemetry, ROUND_CEILINGS);
  EXPECT_NEAR(judgement.data_movement_share, 0.8, 1e-9);
  EXPECT_TRUE(judgement.limited_by == WorkLimit::data_movement);
}

// --- edge / boundary ---

void should_keep_the_estimate_when_the_serial_node_does_not_match() {
  report_fixture::SyntheticRun synthetic_run;
  report_fixture::populate_two_stream_run(synthetic_run);
  give_both_nodes_work(synthetic_run);
  EXPECT_TRUE(work_of_node(synthetic_run.gpu_timeline, 0, synthetic_run.serial_telemetry).bytes_read == 40'000);
}

void should_not_judge_a_node_inside_a_stage_timed_run() {
  report_fixture::SyntheticRun synthetic_run;
  report_fixture::populate_two_stage_run(synthetic_run);
  give_both_nodes_work(synthetic_run);
  const WorkJudgement judgement =
      judge_node(synthetic_run.gpu_timeline, 0, synthetic_run.serial_telemetry, ROUND_CEILINGS);
  EXPECT_TRUE(judgement.limited_by == WorkLimit::not_judged);
}

// --- negative ---

void should_give_no_work_to_a_node_that_launched_no_kernel() {
  report_fixture::SyntheticRun synthetic_run;
  report_fixture::populate_two_stream_run(synthetic_run);
  give_both_nodes_work(synthetic_run);
  const auto& timeline_nodes = synthetic_run.timeline_graph.nodes;
  synthetic_run.serial_telemetry.attributed_nodes = {
      {&timeline_nodes[0], 100.0, 0, 64, {one_launch()}}, {&timeline_nodes[1], 1.0, 0, 64, {}}};
  EXPECT_TRUE(work_of_node(synthetic_run.gpu_timeline, 0, synthetic_run.serial_telemetry).bytes_read == 40'000);
  EXPECT_TRUE(work_of_node(synthetic_run.gpu_timeline, 1, synthetic_run.serial_telemetry).bytes_read == 0);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_judge_a_node_against_its_own_gpu_window),
      TEST_CASE(should_sum_a_stage_tasks_members),
      TEST_CASE(should_keep_the_estimate_when_the_serial_node_does_not_match),
      TEST_CASE(should_not_judge_a_node_inside_a_stage_timed_run),
      TEST_CASE(should_give_no_work_to_a_node_that_launched_no_kernel),
  });
}
