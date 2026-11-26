// Component tests for the HTML report: the full document is rendered from a
// synthetic measurement (no GPU, no filesystem) and checked for the Spark
// views and their invariants.

#include "report_fixture.h"
#include "wired_memory_fixture.h"
#include "report_writer.h"
#include "test_harness.h"

#include <string>

namespace {

using report_fixture::SyntheticRun;

std::string render_synthetic_document(SyntheticRun& synthetic_run,
                                      const std::string& report_title) {
  const mlx_perf::ReportInputs report_inputs{
      report_title, synthetic_run.timeline_graph, synthetic_run.gpu_timeline,
      synthetic_run.serial_telemetry, synthetic_run.job_totals};
  return mlx_perf::build_report_document(report_inputs, "{\"nodes\": []}");
}

std::size_t count_occurrences(const std::string& text, const std::string& needle) {
  std::size_t occurrence_count = 0;
  for (std::size_t found_at = text.find(needle); found_at != std::string::npos;
       found_at = text.find(needle, found_at + needle.size())) {
    ++occurrence_count;
  }
  return occurrence_count;
}

// --- positive ---

void should_render_every_spark_view() {
  SyntheticRun synthetic_run;
  report_fixture::populate_two_stream_run(synthetic_run);
  const std::string document = render_synthetic_document(synthetic_run, "Run");
  for (const char* section_id :
       {"id=\"job\"", "id=\"timeline\"", "id=\"tasks\"", "id=\"dag\"",
        "id=\"method\""}) {
    EXPECT_TRUE(document.find(section_id) != std::string::npos);
  }
  EXPECT_TRUE(document.find("<table class=\"sortable\">") != std::string::npos);
  EXPECT_TRUE(document.find("<svg class=\"gantt\"") != std::string::npos);
}

void should_embed_one_graph_view_with_its_document_and_scripts_once() {
  SyntheticRun synthetic_run;
  report_fixture::populate_two_stream_run(synthetic_run);
  const std::string document = render_synthetic_document(synthetic_run, "Run");
  EXPECT_TRUE(count_occurrences(document, "class=\"graph-view\"") == 1);
  EXPECT_TRUE(document.find("id=\"graph-document-report\">{\"nodes\": []}</script>") !=
              std::string::npos);
  EXPECT_TRUE(count_occurrences(document, "graph view: mounting and interaction") == 1);
  EXPECT_TRUE(document.find(".gv-toolbar") < document.find("</head>"));
}

void should_render_one_task_row_per_node_with_a_stacked_bar() {
  SyntheticRun synthetic_run;
  report_fixture::populate_two_stream_run(synthetic_run);
  const std::string document = render_synthetic_document(synthetic_run, "Run");
  // One bar per task row plus one aggregate bar in the job summary.
  EXPECT_TRUE(count_occurrences(document, "class=\"stacked-bar\"") == 3);
}

void should_draw_stall_and_dependency_arrow_for_the_join() {
  SyntheticRun synthetic_run;
  report_fixture::populate_two_stream_run(synthetic_run);
  const std::string document = render_synthetic_document(synthetic_run, "Run");
  EXPECT_TRUE(count_occurrences(document, "class=\"dependency-arrow\"") == 1);
  EXPECT_TRUE(document.find("<rect class=\"slice-stalled\"") != std::string::npos);
  EXPECT_TRUE(document.find("wait_on_dependency") != std::string::npos);
}

void should_draw_the_wired_memory_lane_under_the_timeline() {
  SyntheticRun synthetic_run;
  report_fixture::populate_two_stream_run(synthetic_run);
  synthetic_run.serial_telemetry.wired_memory_samples = wired_memory_fixture::release_then_unwire();
  const std::string document = render_synthetic_document(synthetic_run, "Run");
  const std::size_t timeline_at = document.find("id=\"timeline\"");
  const std::size_t wired_memory_at = document.find("id=\"wired-memory\"");
  EXPECT_TRUE(wired_memory_at != std::string::npos && wired_memory_at > timeline_at);
  EXPECT_TRUE(document.find("Sampled in the serial pass") != std::string::npos);
}

// --- edge / boundary ---

void should_leave_the_wired_memory_lane_out_without_samples() {
  SyntheticRun synthetic_run;
  report_fixture::populate_two_stream_run(synthetic_run);
  synthetic_run.serial_telemetry.wired_memory_samples.clear();
  EXPECT_TRUE(render_synthetic_document(synthetic_run, "Run").find("id=\"wired-memory\"") == std::string::npos);
}

void should_render_empty_measurement_without_nan_or_infinity() {
  SyntheticRun empty_run;
  const std::string document = render_synthetic_document(empty_run, "Empty");
  const std::string visible_page = report_fixture::remove_script_elements(document);
  EXPECT_TRUE(visible_page.find("nan") == std::string::npos);
  EXPECT_TRUE(visible_page.find("inf\"") == std::string::npos);
  EXPECT_TRUE(document.find("</html>") != std::string::npos);
}

void should_show_dash_when_serial_pass_has_no_matching_node() {
  SyntheticRun synthetic_run;
  report_fixture::populate_two_stream_run(synthetic_run);
  const std::string document = render_synthetic_document(synthetic_run, "Run");
  // Two nodes × (Serial, Mem added, Launches).
  EXPECT_TRUE(count_occurrences(document, ">—</td>") == 6);
}

void should_flag_a_node_whose_launch_cannot_fill_the_gpu() {
  SyntheticRun synthetic_run;
  report_fixture::populate_two_stream_run(synthetic_run);
  mlx_perf::KernelDispatch four_threadgroups;
  four_threadgroups.grid_unit = mlx_perf::DispatchGridUnit::threads;
  four_threadgroups.grid = {128, 1, 1};
  four_threadgroups.threads_per_threadgroup = {32, 1, 1};
  const auto& timeline_nodes = synthetic_run.timeline_graph.nodes;
  synthetic_run.serial_telemetry.attributed_nodes = {
      {&timeline_nodes[0], 40.0, 0, 64, {four_threadgroups}}, {&timeline_nodes[1], 9.0, 0, 64, {}}};
  synthetic_run.serial_telemetry.gpu_device_facts = {"Apple M5 Pro", 16};
  const std::string document = render_synthetic_document(synthetic_run, "Run");
  EXPECT_TRUE(document.find("⚠ 4 threadgroups · 4 of 16 cores") != std::string::npos);
  EXPECT_TRUE(count_occurrences(document, "underfills-gpu\"") == 1);
}

// --- negative ---

void should_escape_hostile_titles_and_node_labels() {
  SyntheticRun synthetic_run;
  report_fixture::populate_two_stream_run(synthetic_run);
  const std::string document =
      render_synthetic_document(synthetic_run, "<script>alert(1)</script>");
  EXPECT_TRUE(document.find("<script>alert(1)") == std::string::npos);
  // The fixture's join is labeled "B<x>": it must never reach the page raw.
  EXPECT_TRUE(document.find("B<x>") == std::string::npos);
  EXPECT_TRUE(document.find("B&lt;x&gt;") != std::string::npos);
}

// A per-stage measurement renders the stage table (one row per stage task)
// and stage bars on the Gantt, never per-node rows of zeros.
void should_render_stage_rows_and_stage_bars_for_a_per_stage_timeline() {
  SyntheticRun synthetic_run;
  report_fixture::populate_two_stage_run(synthetic_run);
  const std::string document = render_synthetic_document(synthetic_run, "staged");
  EXPECT_TRUE(document.find("Tasks (one per stage and stream)") != std::string::npos);
  EXPECT_TRUE(document.find("one bar per stage") != std::string::npos);
  EXPECT_TRUE(document.find("layer 1 (stream 0, 1 nodes)") != std::string::npos);
  EXPECT_TRUE(document.find("This report was timed per stage") != std::string::npos);
  // One summary bar plus one per stage task.
  EXPECT_TRUE(count_occurrences(document, "class=\"stacked-bar\"") == 3);
  EXPECT_TRUE(count_occurrences(document, "class=\"dependency-arrow\"") == 1);
  EXPECT_TRUE(document.find("Where stage time went") != std::string::npos);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_render_every_spark_view),
      TEST_CASE(should_embed_one_graph_view_with_its_document_and_scripts_once),
      TEST_CASE(should_draw_the_wired_memory_lane_under_the_timeline),
      TEST_CASE(should_leave_the_wired_memory_lane_out_without_samples),
      TEST_CASE(should_render_one_task_row_per_node_with_a_stacked_bar),
      TEST_CASE(should_draw_stall_and_dependency_arrow_for_the_join),
      TEST_CASE(should_render_empty_measurement_without_nan_or_infinity),
      TEST_CASE(should_show_dash_when_serial_pass_has_no_matching_node),
      TEST_CASE(should_flag_a_node_whose_launch_cannot_fill_the_gpu),
      TEST_CASE(should_escape_hostile_titles_and_node_labels),
      TEST_CASE(should_render_stage_rows_and_stage_bars_for_a_per_stage_timeline),
  });
}
