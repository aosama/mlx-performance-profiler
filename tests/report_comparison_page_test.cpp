// Component tests for the eager-vs-compiled comparison page, rendered from
// synthetic measurements (no GPU).

#include "report_comparison_page.h"
#include "report_fixture.h"
#include "test_harness.h"

#include <string>

namespace {

using namespace mlx_perf;

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
      "Job", {eager_measurement, "{\"variant_name\": \"eager\"}", "eager.html"},
      {compiled_measurement, "", "compiled.html"});
  EXPECT_TRUE(document.find("-50.0%") != std::string::npos);
  EXPECT_TRUE(count_occurrences(document, "<svg class=\"gantt\"") == 2);
  EXPECT_TRUE(document.find("id=\"primitives\"") != std::string::npos);
  EXPECT_TRUE(document.find("href=\"compiled.html\"") != std::string::npos);
  EXPECT_TRUE(document.find("{\"variant_name\": \"eager\"}") != std::string::npos);
}

void should_give_each_variant_its_own_graph_view_and_load_the_scripts_once() {
  VariantMeasurement eager_measurement;
  VariantMeasurement compiled_measurement;
  populate_variant(eager_measurement, "eager", 200);
  populate_variant(compiled_measurement, "compiled", 100);
  const std::string document = build_comparison_document(
      "Job", {eager_measurement, "{\"variant_name\": \"eager\"}", "e.html"},
      {compiled_measurement, "{\"variant_name\": \"compiled\"}", "c.html"});
  EXPECT_TRUE(count_occurrences(document, "class=\"graph-view\"") == 2);
  EXPECT_TRUE(document.find("id=\"graph-document-baseline\">{\"variant_name\": \"eager\"}") !=
              std::string::npos);
  EXPECT_TRUE(document.find("id=\"graph-document-candidate\">{\"variant_name\": \"compiled\"}") !=
              std::string::npos);
  EXPECT_TRUE(count_occurrences(document, "graph view: mounting and interaction") == 1);
}

void should_open_with_the_callers_verdict_and_subtitle() {
  VariantMeasurement stock_measurement;
  VariantMeasurement custom_measurement;
  populate_variant(stock_measurement, "stock", 200);
  populate_variant(custom_measurement, "custom", 100);
  const std::string document = build_comparison_document(
      "Job", {stock_measurement, "", "stock.html"}, {custom_measurement, "", "custom.html"},
      {"Quantized projection, two kernels", "<p class=\"verdict faster\">VERDICT</p>"});
  EXPECT_TRUE(document.find("Quantized projection, two kernels") != std::string::npos);
  EXPECT_TRUE(document.find("id=\"verdict\"") < document.find("id=\"compare\""));
  EXPECT_TRUE(document.find("VERDICT") != std::string::npos);
}

// Without a verdict (no rounds to judge) the page has no verdict section, and
// the default subtitle names both implementations.
void should_percent_encode_report_links_with_special_characters() {
  VariantMeasurement stock_measurement;
  VariantMeasurement custom_measurement;
  populate_variant(stock_measurement, "stock", 200);
  populate_variant(custom_measurement, "v2#fast", 100);
  const std::string document = build_comparison_document(
      "Job", {stock_measurement, "", "base_stock.html"},
      {custom_measurement, "", "base_v2#fast.html"});
  EXPECT_TRUE(document.find("href=\"base_v2%23fast.html\"") != std::string::npos);
}

void should_omit_the_verdict_and_the_fused_column_when_there_is_nothing_to_show() {
  VariantMeasurement stock_measurement;
  VariantMeasurement custom_measurement;
  populate_variant(stock_measurement, "stock", 200);
  populate_variant(custom_measurement, "custom", 100);
  const std::string document = build_comparison_document(
      "Job", {stock_measurement, "", "stock.html"}, {custom_measurement, "", "custom.html"});
  EXPECT_TRUE(document.find("id=\"verdict\"") == std::string::npos);
  EXPECT_TRUE(document.find("Same job, two implementations: stock vs custom") !=
              std::string::npos);
  EXPECT_TRUE(document.find("<th>Fuses</th>") == std::string::npos);
}

// A stage-timed variant has no per-node GPU time: its primitive mix must
// come from the serial pass (labeled), and the page must say the two
// variants were timed at different granularities.
void should_flag_mixed_granularities_and_tally_stage_timed_primitives_from_serial() {
  VariantMeasurement node_timed;
  VariantMeasurement stage_timed;
  populate_variant(node_timed, "eager", 200);
  report_fixture::populate_two_stage_run(stage_timed);
  stage_timed.variant_name = "compiled";
  stage_timed.serial_telemetry.attributed_nodes = {
      {&stage_timed.timeline_graph.nodes[0], 33.0, 0, 64},
      {&stage_timed.timeline_graph.nodes[1], 4.0, 0, 64}};
  const std::string metrics_html = render_comparison_metrics(node_timed, stage_timed);
  EXPECT_TRUE(metrics_html.find("eager timed per-node, compiled timed per-stage") !=
              std::string::npos);
  EXPECT_TRUE(metrics_html.find("not like for like") != std::string::npos);
  const std::string document = build_comparison_document(
      "Job", {node_timed, "", "eager.html"}, {stage_timed, "", "compiled.html"});
  EXPECT_TRUE(document.find("compiled serial Σ (stage-timed)") != std::string::npos);
  EXPECT_TRUE(document.find("data-sort-value=\"33\"") != std::string::npos);
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
  EXPECT_TRUE(report_fixture::remove_script_elements(document).find("inf") == std::string::npos);
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
      TEST_CASE(should_give_each_variant_its_own_graph_view_and_load_the_scripts_once),
      TEST_CASE(should_open_with_the_callers_verdict_and_subtitle),
      TEST_CASE(should_percent_encode_report_links_with_special_characters),
      TEST_CASE(should_omit_the_verdict_and_the_fused_column_when_there_is_nothing_to_show),
      TEST_CASE(should_flag_mixed_granularities_and_tally_stage_timed_primitives_from_serial),
      TEST_CASE(should_leave_a_bare_compiled_name_and_single_primitive_sensible),
      TEST_CASE(should_show_dash_for_change_when_baseline_is_zero),
      TEST_CASE(should_not_describe_non_compiled_primitives_as_fused),
      TEST_CASE(should_escape_hostile_variant_names),
  });
}
