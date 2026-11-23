// Component tests for the eager-vs-compiled comparison page, rendered from
// synthetic measurements (no GPU).

#include "report_comparison_page.h"
#include "report_fixture.h"
#include "test_harness.h"

#include <string>

namespace {

using namespace mlx_plan;

std::size_t count_occurrences(const std::string& text, const std::string& needle) {
  std::size_t occurrence_count = 0;
  for (std::size_t found_at = text.find(needle); found_at != std::string::npos;
       found_at = text.find(needle, found_at + needle.size())) {
    ++occurrence_count;
  }
  return occurrence_count;
}

void populate_variant(VariantMeasurement& variant_measurement,
                      const std::string& variant_name,
                      double job_wall_microseconds) {
  report_fixture::populate_two_stream_run(variant_measurement);
  variant_measurement.variant_name = variant_name;
  variant_measurement.job_totals.job_wall_microseconds = job_wall_microseconds;
}

// --- positive ---

void should_split_a_compiled_name_into_its_fused_primitives() {
  EXPECT_TRUE(describe_fused_primitives("CompiledSigmoidMultiply") ==
              "Sigmoid · Multiply");
  EXPECT_TRUE(describe_fused_primitives("CompiledExpBroadcastMultiplySinAdd") ==
              "Exp · Broadcast · Multiply · Sin · Add");
}

void should_render_metrics_plans_shared_timelines_and_primitive_mix() {
  VariantMeasurement eager_measurement;
  VariantMeasurement compiled_measurement;
  populate_variant(eager_measurement, "eager", 200);
  populate_variant(compiled_measurement, "compiled", 100);
  const std::string document = build_comparison_document(
      "Job", {eager_measurement, "<svg id=\"a\"></svg>", "eager.html"},
      {compiled_measurement, "", "compiled.html"});
  EXPECT_TRUE(document.find("-50.0%") != std::string::npos);
  EXPECT_TRUE(count_occurrences(document, "<svg class=\"gantt\"") == 2);
  EXPECT_TRUE(document.find("id=\"primitives\"") != std::string::npos);
  EXPECT_TRUE(document.find("href=\"compiled.html\"") != std::string::npos);
  EXPECT_TRUE(document.find("<svg id=\"a\">") != std::string::npos);
}

// --- edge / boundary ---

void should_leave_a_bare_compiled_name_and_single_primitive_sensible() {
  EXPECT_TRUE(describe_fused_primitives("Compiled").empty());
  EXPECT_TRUE(describe_fused_primitives("CompiledExp") == "Exp");
}

void should_show_dash_for_change_when_baseline_is_zero() {
  VariantMeasurement eager_measurement;
  VariantMeasurement compiled_measurement;
  populate_variant(eager_measurement, "eager", 0);
  populate_variant(compiled_measurement, "compiled", 100);
  const std::string document = build_comparison_document(
      "Job", {eager_measurement, "", "e.html"}, {compiled_measurement, "", "c.html"});
  EXPECT_TRUE(document.find("inf") == std::string::npos);
  EXPECT_TRUE(document.find(">—</td>") != std::string::npos);
}

// --- negative ---

void should_not_describe_non_compiled_primitives_as_fused() {
  EXPECT_TRUE(describe_fused_primitives("Add").empty());
  EXPECT_TRUE(describe_fused_primitives("").empty());
  EXPECT_TRUE(describe_fused_primitives("compiledExp").empty());
}

void should_escape_hostile_variant_names() {
  VariantMeasurement eager_measurement;
  VariantMeasurement compiled_measurement;
  populate_variant(eager_measurement, "<b>eager</b>", 100);
  populate_variant(compiled_measurement, "compiled", 100);
  const std::string document = build_comparison_document(
      "Job", {eager_measurement, "", "e.html"}, {compiled_measurement, "", "c.html"});
  EXPECT_TRUE(document.find("<b>eager</b>") == std::string::npos);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_split_a_compiled_name_into_its_fused_primitives),
      TEST_CASE(should_render_metrics_plans_shared_timelines_and_primitive_mix),
      TEST_CASE(should_leave_a_bare_compiled_name_and_single_primitive_sensible),
      TEST_CASE(should_show_dash_for_change_when_baseline_is_zero),
      TEST_CASE(should_not_describe_non_compiled_primitives_as_fused),
      TEST_CASE(should_escape_hostile_variant_names),
  });
}
