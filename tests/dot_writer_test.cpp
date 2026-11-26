// Component tests for dot_writer: the DOT text for a synthetic measurement,
// per node and per stage (no GPU, no Graphviz).

#include "dot_writer.h"
#include "report_fixture.h"
#include "test_harness.h"

#include <string>

namespace {

using report_fixture::SyntheticRun;

// --- positive ---

void should_label_nodes_with_their_measured_compute_and_class() {
  SyntheticRun synthetic_run;
  report_fixture::populate_two_stream_run(synthetic_run);
  const std::string plan_dot = mlx_perf::write_plan_dot(
      synthetic_run.timeline_graph, synthetic_run.gpu_timeline, "plan");
  EXPECT_TRUE(plan_dot.find("wait_on_dependency") != std::string::npos);
  EXPECT_TRUE(plan_dot.find("penwidth=2.5") != std::string::npos);
  EXPECT_TRUE(plan_dot.find("subgraph") == std::string::npos);
}

// Spark boxes each stage in its DAG view; a per-stage pass does the same and
// never prints a per-node number it did not measure.
void should_cluster_nodes_by_stage_and_name_the_stage() {
  SyntheticRun synthetic_run;
  report_fixture::populate_two_stage_run(synthetic_run);
  const std::string plan_dot = mlx_perf::write_plan_dot(
      synthetic_run.timeline_graph, synthetic_run.gpu_timeline, "plan");
  EXPECT_TRUE(plan_dot.find("subgraph \"cluster_stage_0\"") != std::string::npos);
  EXPECT_TRUE(plan_dot.find("subgraph \"cluster_stage_1\"") != std::string::npos);
  EXPECT_TRUE(plan_dot.find("stage: layer 1") != std::string::npos);
  EXPECT_TRUE(plan_dot.find("gpu ") == std::string::npos);
  // The stalled join stage outlines its member.
  EXPECT_TRUE(plan_dot.find("penwidth=2.5") != std::string::npos);
}

// --- edge / boundary ---

void should_write_a_valid_empty_digraph() {
  const mlx_perf::CapturedGraph empty_graph;
  const mlx_perf::GpuTimeline empty_timeline;
  const std::string plan_dot = mlx_perf::write_plan_dot(empty_graph, empty_timeline, "empty");
  EXPECT_TRUE(plan_dot.rfind("digraph \"empty\" {", 0) == 0);
  EXPECT_TRUE(plan_dot.find("}\n") != std::string::npos);
}

// --- negative ---

void should_escape_quotes_in_stage_names() {
  SyntheticRun synthetic_run;
  report_fixture::populate_two_stage_run(synthetic_run);
  synthetic_run.gpu_timeline.stages[0].stage_name = "layer \"0\"";
  const std::string plan_dot = mlx_perf::write_plan_dot(
      synthetic_run.timeline_graph, synthetic_run.gpu_timeline, "plan");
  EXPECT_TRUE(plan_dot.find("label=\"layer \\\"0\\\"\"") != std::string::npos);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_label_nodes_with_their_measured_compute_and_class),
      TEST_CASE(should_cluster_nodes_by_stage_and_name_the_stage),
      TEST_CASE(should_write_a_valid_empty_digraph),
      TEST_CASE(should_escape_quotes_in_stage_names),
  });
}
