// Component tests for the HTML report: the full document is rendered from a
// synthetic measurement (no GPU, no filesystem) and checked for the Spark
// views and their invariants.

#include "report_fixture.h"
#include "report_writer.h"
#include "test_harness.h"

#include <string>

namespace {

using report_fixture::SyntheticRun;

std::string render_synthetic_document(SyntheticRun& synthetic_run,
                                      const std::string& report_title) {
  const mlx_plan::ReportInputs report_inputs{
      report_title, synthetic_run.timeline_graph, synthetic_run.gpu_timeline,
      synthetic_run.serial_telemetry, synthetic_run.job_totals};
  return mlx_plan::build_report_document(report_inputs, "", "digraph {}");
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

// --- edge / boundary ---

void should_render_empty_measurement_without_nan_or_infinity() {
  SyntheticRun empty_run;
  const std::string document = render_synthetic_document(empty_run, "Empty");
  EXPECT_TRUE(document.find("nan") == std::string::npos);
  EXPECT_TRUE(document.find("inf\"") == std::string::npos);
  EXPECT_TRUE(document.find("</html>") != std::string::npos);
}

void should_show_dash_when_serial_pass_has_no_matching_node() {
  SyntheticRun synthetic_run;
  report_fixture::populate_two_stream_run(synthetic_run);
  const std::string document = render_synthetic_document(synthetic_run, "Run");
  // Two nodes × (Serial, Mem added).
  EXPECT_TRUE(count_occurrences(document, ">—</td>") == 4);
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
      TEST_CASE(should_render_one_task_row_per_node_with_a_stacked_bar),
      TEST_CASE(should_draw_stall_and_dependency_arrow_for_the_join),
      TEST_CASE(should_render_empty_measurement_without_nan_or_infinity),
      TEST_CASE(should_show_dash_when_serial_pass_has_no_matching_node),
      TEST_CASE(should_escape_hostile_titles_and_node_labels),
      TEST_CASE(should_render_stage_rows_and_stage_bars_for_a_per_stage_timeline),
  });
}
