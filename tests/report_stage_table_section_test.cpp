// Component tests for the per-stage task table: rendered from a synthetic
// per-stage measurement (no GPU) and checked for rows, sums and the op mix.

#include "report_fixture.h"
#include "report_stage_table_section.h"
#include "test_harness.h"

#include <string>

namespace {

using report_fixture::SyntheticRun;

std::string render_stage_table(SyntheticRun& synthetic_run) {
  const mlx_perf::ReportInputs report_inputs{
      "staged", synthetic_run.timeline_graph, synthetic_run.gpu_timeline,
      synthetic_run.serial_telemetry, synthetic_run.job_totals};
  return mlx_perf::render_stage_table_section(report_inputs);
}

// --- positive ---

void should_render_one_row_per_stage_task_with_its_class() {
  SyntheticRun synthetic_run;
  report_fixture::populate_two_stage_run(synthetic_run);
  const std::string section = render_stage_table(synthetic_run);
  EXPECT_TRUE(section.find("<td>layer 0</td>") != std::string::npos);
  EXPECT_TRUE(section.find("<td>layer 1</td>") != std::string::npos);
  EXPECT_TRUE(section.find("MatMul ×1") != std::string::npos);
  EXPECT_TRUE(section.find("wait_on_dependency") != std::string::npos);
}

void should_sum_serial_durations_over_the_stage_members() {
  SyntheticRun synthetic_run;
  report_fixture::populate_two_stage_run(synthetic_run);
  synthetic_run.serial_telemetry.attributed_nodes = {
      {&synthetic_run.timeline_graph.nodes[0], 42.0, 128, 64},
      {&synthetic_run.timeline_graph.nodes[1], 7.0, 0, 64}};
  const std::string section = render_stage_table(synthetic_run);
  EXPECT_TRUE(section.find("data-sort-value=\"42\"") != std::string::npos);
  EXPECT_TRUE(section.find(">—</td>") == std::string::npos);
}

void should_rank_ops_by_count_then_name_and_truncate() {
  SyntheticRun synthetic_run;
  report_fixture::populate_two_stream_run(synthetic_run);
  synthetic_run.timeline_graph.nodes[1].op_name = "MatMul";
  const mlx_perf::GpuTimeline& gpu_timeline = synthetic_run.gpu_timeline;
  EXPECT_TRUE(mlx_perf::summarize_op_mix(gpu_timeline, {0, 1}, 4) == "MatMul ×2");
  synthetic_run.timeline_graph.nodes[1].op_name = "Add";
  EXPECT_TRUE(mlx_perf::summarize_op_mix(gpu_timeline, {0, 1}, 4) == "Add ×1, MatMul ×1");
  EXPECT_TRUE(mlx_perf::summarize_op_mix(gpu_timeline, {0, 1}, 1) == "Add ×1, …");
}

// --- edge / boundary ---

void should_render_an_empty_table_for_no_stages() {
  SyntheticRun synthetic_run;
  synthetic_run.gpu_timeline.granularity = mlx_perf::TimelineGranularity::per_stage;
  const std::string section = render_stage_table(synthetic_run);
  EXPECT_TRUE(section.find("<tbody></tbody>") != std::string::npos);
  EXPECT_TRUE(mlx_perf::summarize_op_mix(synthetic_run.gpu_timeline, {}, 4).empty());
}

// --- negative ---

// A stage whose members fail the serial join shows "—", never a partial sum.
void should_show_dashes_when_the_serial_pass_does_not_match() {
  SyntheticRun synthetic_run;
  report_fixture::populate_two_stage_run(synthetic_run);
  const std::string section = render_stage_table(synthetic_run);
  EXPECT_TRUE(section.find(">—</td>") != std::string::npos);
}

void should_escape_hostile_stage_names() {
  SyntheticRun synthetic_run;
  report_fixture::populate_two_stage_run(synthetic_run);
  synthetic_run.gpu_timeline.stages[0].stage_name = "<script>alert(1)</script>";
  const std::string section = render_stage_table(synthetic_run);
  EXPECT_TRUE(section.find("<script>") == std::string::npos);
  EXPECT_TRUE(section.find("&lt;script&gt;") != std::string::npos);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_render_one_row_per_stage_task_with_its_class),
      TEST_CASE(should_sum_serial_durations_over_the_stage_members),
      TEST_CASE(should_rank_ops_by_count_then_name_and_truncate),
      TEST_CASE(should_render_an_empty_table_for_no_stages),
      TEST_CASE(should_show_dashes_when_the_serial_pass_does_not_match),
      TEST_CASE(should_escape_hostile_stage_names),
  });
}
