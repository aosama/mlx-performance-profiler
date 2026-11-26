// Component tests for live_view_renderer: the four tab fragments, from
// synthetic measurements (no GPU).

#include "live_view_renderer.h"
#include "report_fixture.h"
#include "test_harness.h"

#include <string>

namespace {

using namespace mlx_perf;

bool contains(const std::string& text, const std::string& needle) {
  return text.find(needle) != std::string::npos;
}

// --- positive ---

void should_open_the_job_tab_of_a_pair_with_the_verdict() {
  PairedMeasurement paired_measurement;
  report_fixture::populate_pair(paired_measurement);
  const RenderedViews rendered_views = render_live_views("Job", paired_measurement);
  EXPECT_TRUE(rendered_views.job_html.rfind("<section><h2>Verdict</h2>", 0) == 0);
  EXPECT_TRUE(contains(rendered_views.job_html, "class=\"verdict"));
  // After the verdict: the single-figure table, then each variant's views.
  EXPECT_TRUE(rendered_views.job_html.find("Change (not judged)") >
              rendered_views.job_html.find("class=\"verdict"));
  for (const std::string* tab_html : {&rendered_views.stages_html, &rendered_views.tasks_html,
                                      &rendered_views.memory_html}) {
    EXPECT_TRUE(contains(*tab_html, ">stock<") && contains(*tab_html, ">custom<"));
  }
}

// --- edge / boundary ---

void should_judge_memory_and_note_pressure_in_the_live_verdict() {
  PairedMeasurement paired_measurement;
  report_fixture::populate_pair_with_memory(paired_measurement);
  const RenderedViews rendered_views = render_live_views("Job", paired_measurement);
  EXPECT_TRUE(rendered_views.job_html.find("Memory per run") != std::string::npos);
  EXPECT_TRUE(rendered_views.job_html.find("custom crossed the wire limit in 1 of 12 rounds") != std::string::npos);
}

void should_render_a_single_variant_without_comparison_or_verdict() {
  VariantMeasurement variant_measurement;
  report_fixture::populate_two_stream_run(variant_measurement);
  variant_measurement.variant_name = "only";
  const RenderedViews rendered_views = render_live_views("Job", {&variant_measurement});
  EXPECT_TRUE(!contains(rendered_views.job_html, "Verdict"));
  EXPECT_TRUE(!contains(rendered_views.job_html, "Change (not judged)"));
  EXPECT_TRUE(contains(rendered_views.job_html, ">only<"));
}

// --- negative: a variant name is text, never markup ---

void should_escape_variant_names_in_every_tab() {
  PairedMeasurement paired_measurement;
  report_fixture::populate_pair(paired_measurement);
  paired_measurement.candidate.variant_name = "<script>x</script>";
  const RenderedViews rendered_views = render_live_views("Job", paired_measurement);
  for (const std::string* tab_html : {&rendered_views.job_html, &rendered_views.stages_html,
                                      &rendered_views.tasks_html,
                                      &rendered_views.memory_html}) {
    EXPECT_TRUE(!contains(*tab_html, "<script>x</script>"));
  }
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_open_the_job_tab_of_a_pair_with_the_verdict),
      TEST_CASE(should_judge_memory_and_note_pressure_in_the_live_verdict),
      TEST_CASE(should_render_a_single_variant_without_comparison_or_verdict),
      TEST_CASE(should_escape_variant_names_in_every_tab),
  });
}
