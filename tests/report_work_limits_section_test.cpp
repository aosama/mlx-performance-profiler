// Component tests for report_work_limits_section: the "Data and arithmetic"
// section's ceilings, rows and verdicts, from a synthetic run. No GPU.

#include "report_fixture.h"
#include "report_work_limits_section.h"
#include "test_harness.h"

#include <string>

namespace {

using namespace mlx_perf;

bool contains(const std::string& page_html, const std::string& expected_text) {
  return page_html.find(expected_text) != std::string::npos;
}

std::string render_with(report_fixture::SyntheticRun& synthetic_run, const MachineCeilings& machine_ceilings) {
  const ReportInputs report_inputs{"limits", synthetic_run.timeline_graph, synthetic_run.gpu_timeline,
                                   synthetic_run.serial_telemetry, synthetic_run.job_totals, machine_ceilings};
  return render_work_limits_section(report_inputs);
}

// The producer moved 80,000 bytes (80 µs of its 100 µs at the ceiling).
void give_the_producer_work(report_fixture::SyntheticRun& synthetic_run) {
  synthetic_run.timeline_graph.nodes[0].work.bytes_read = 80'000;
}

const MachineCeilings ROUND_CEILINGS{true, 1e9, 1e12, "bfloat16"};

// --- positive ---

void should_show_the_ceilings_and_a_row_per_node_that_moved_data() {
  report_fixture::SyntheticRun synthetic_run;
  report_fixture::populate_two_stream_run(synthetic_run);
  give_the_producer_work(synthetic_run);
  const std::string section_html = render_with(synthetic_run, ROUND_CEILINGS);
  EXPECT_TRUE(contains(section_html, "id=\"data-and-arithmetic\""));
  EXPECT_TRUE(contains(section_html, "0.9 GiB/s"));
  EXPECT_TRUE(contains(section_html, "1.0 TFLOP/s"));
  EXPECT_TRUE(contains(section_html, "a bfloat16 matrix multiply"));
  EXPECT_TRUE(contains(section_html, "80%"));
  EXPECT_TRUE(contains(section_html, "limit-badge limit-data"));
}

void should_show_one_row_per_stage_task_in_a_stage_timed_run() {
  report_fixture::SyntheticRun synthetic_run;
  report_fixture::populate_two_stage_run(synthetic_run);
  give_the_producer_work(synthetic_run);
  const std::string section_html = render_with(synthetic_run, ROUND_CEILINGS);
  EXPECT_TRUE(contains(section_html, "layer 0"));
  EXPECT_TRUE(contains(section_html, "1 node, stream 1"));
  EXPECT_TRUE(!contains(section_html, "layer 1"));
}

// --- edge / boundary ---

void should_say_not_counted_for_arithmetic_it_does_not_count() {
  report_fixture::SyntheticRun synthetic_run;
  report_fixture::populate_two_stream_run(synthetic_run);
  give_the_producer_work(synthetic_run);
  EXPECT_TRUE(contains(render_with(synthetic_run, ROUND_CEILINGS), "not counted"));
}

void should_mark_neither_when_the_gpu_was_not_kept_busy() {
  report_fixture::SyntheticRun synthetic_run;
  report_fixture::populate_two_stream_run(synthetic_run);
  synthetic_run.timeline_graph.nodes[0].work.bytes_read = 10'000;
  EXPECT_TRUE(contains(render_with(synthetic_run, ROUND_CEILINGS), "limit-badge limit-neither"));
}

void should_show_a_share_above_one_hundred_percent_with_a_capped_bar() {
  report_fixture::SyntheticRun synthetic_run;
  report_fixture::populate_two_stream_run(synthetic_run);
  give_the_producer_work(synthetic_run);
  synthetic_run.timeline_graph.nodes[0].work.is_arithmetic_counted = true;
  synthetic_run.timeline_graph.nodes[0].work.arithmetic_operations = 150e6;
  const std::string section_html = render_with(synthetic_run, ROUND_CEILINGS);
  EXPECT_TRUE(contains(section_html, "150%"));
  EXPECT_TRUE(contains(section_html, "need-arithmetic\" style=\"width:100%\""));
  EXPECT_TRUE(contains(section_html, "limit-badge limit-arithmetic"));
}

// --- negative ---

void should_say_so_when_the_ceilings_were_not_measured() {
  report_fixture::SyntheticRun synthetic_run;
  report_fixture::populate_two_stream_run(synthetic_run);
  give_the_producer_work(synthetic_run);
  const std::string section_html = render_with(synthetic_run, MachineCeilings{});
  EXPECT_TRUE(contains(section_html, "were not measured"));
  EXPECT_TRUE(!contains(section_html, "<table"));
}

void should_escape_a_custom_kernel_name() {
  report_fixture::SyntheticRun synthetic_run;
  report_fixture::populate_two_stream_run(synthetic_run);
  give_the_producer_work(synthetic_run);
  synthetic_run.timeline_graph.nodes[0].kernel_name = "custom_kernel_<script>";
  const std::string section_html = render_with(synthetic_run, ROUND_CEILINGS);
  EXPECT_TRUE(contains(section_html, "custom_kernel_&lt;script&gt;"));
  EXPECT_TRUE(!contains(section_html, "custom_kernel_<script>"));
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_show_the_ceilings_and_a_row_per_node_that_moved_data),
      TEST_CASE(should_show_one_row_per_stage_task_in_a_stage_timed_run),
      TEST_CASE(should_say_not_counted_for_arithmetic_it_does_not_count),
      TEST_CASE(should_mark_neither_when_the_gpu_was_not_kept_busy),
      TEST_CASE(should_show_a_share_above_one_hundred_percent_with_a_capped_bar),
      TEST_CASE(should_say_so_when_the_ceilings_were_not_measured),
      TEST_CASE(should_escape_a_custom_kernel_name),
  });
}
