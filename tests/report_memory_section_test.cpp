// Unit tests for the memory views (no GPU).

#include "report_memory_section.h"
#include "report_fixture.h"
#include "test_harness.h"

#include <string>

namespace {

using namespace mlx_perf;

bool contains(const std::string& text, const std::string& needle) {
  return text.find(needle) != std::string::npos;
}

void add_serial_node(VariantMeasurement& variant_measurement, int graph_position,
                     std::size_t memory_added_bytes) {
  NodeTelemetry node_telemetry;
  node_telemetry.node = &variant_measurement.timeline_graph.nodes[graph_position];
  node_telemetry.memory_added_bytes = memory_added_bytes;
  node_telemetry.output_bytes = 64;
  variant_measurement.serial_telemetry.attributed_nodes.push_back(node_telemetry);
}

// --- positive ---

void should_rank_nodes_by_memory_added_largest_first() {
  VariantMeasurement variant_measurement;
  report_fixture::populate_two_stream_run(variant_measurement);
  add_serial_node(variant_measurement, 0, 1024);
  add_serial_node(variant_measurement, 1, 4096);
  const std::string section_html = render_memory_section(variant_measurement);
  EXPECT_TRUE(section_html.find("4.0 KiB") < section_html.find("1.0 KiB"));
  EXPECT_TRUE(contains(section_html, "width:100%"));
}

void should_draw_three_memory_lines_from_samples() {
  const std::string chart_html = render_memory_timeline(
      {{0.0, 100, 200, 50}, {1.0, 150, 200, 60}, {2.0, 120, 220, 70}});
  EXPECT_TRUE(contains(chart_html, "line-compute"));
  EXPECT_TRUE(contains(chart_html, "line-scheduling"));
  EXPECT_TRUE(contains(chart_html, "line-result"));
}

// --- edge / boundary ---

void should_ask_for_patience_with_fewer_than_two_samples() {
  EXPECT_TRUE(contains(render_memory_timeline({}), "Collecting"));
  EXPECT_TRUE(contains(render_memory_timeline({{0.0, 1, 1, 1}}), "Collecting"));
}

void should_render_empty_ranking_without_bars() {
  VariantMeasurement variant_measurement;
  const std::string section_html = render_memory_section(variant_measurement);
  EXPECT_TRUE(!contains(section_html, "stacked-bar"));
  EXPECT_TRUE(contains(section_html, "</table>"));
}

// --- negative ---

void should_not_divide_by_zero_when_every_sample_is_zero() {
  const std::string chart_html =
      render_memory_timeline({{0.0, 0, 0, 0}, {0.0, 0, 0, 0}});
  EXPECT_TRUE(!contains(chart_html, "nan"));
  EXPECT_TRUE(!contains(chart_html, "inf"));
}

void should_not_divide_by_zero_when_no_node_added_memory() {
  VariantMeasurement variant_measurement;
  report_fixture::populate_two_stream_run(variant_measurement);
  add_serial_node(variant_measurement, 0, 0);
  const std::string section_html = render_memory_section(variant_measurement);
  EXPECT_TRUE(!contains(section_html, "nan"));
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_rank_nodes_by_memory_added_largest_first),
      TEST_CASE(should_draw_three_memory_lines_from_samples),
      TEST_CASE(should_ask_for_patience_with_fewer_than_two_samples),
      TEST_CASE(should_render_empty_ranking_without_bars),
      TEST_CASE(should_not_divide_by_zero_when_every_sample_is_zero),
      TEST_CASE(should_not_divide_by_zero_when_no_node_added_memory),
  });
}
