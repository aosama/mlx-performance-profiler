// Component tests for report_verdict_section: the verdict sentences and the
// metric table, from synthetic comparisons (no GPU).

#include "report_fixture.h"
#include "report_verdict_section.h"
#include "test_harness.h"

#include <string>

namespace {

using namespace mlx_perf;

bool contains(const std::string& text, const std::string& needle) {
  return text.find(needle) != std::string::npos;
}

// --- positive ---

void should_state_which_is_faster_by_how_much_and_against_the_spread() {
  const PairedComparison paired_comparison = compare_paired_samples({100, 104, 102}, {90, 93, 91});
  const std::string verdict = describe_verdict("stock", "custom", paired_comparison);
  EXPECT_TRUE(contains(verdict, "custom is 10.8% faster than stock"));
  EXPECT_TRUE(contains(verdict, "(11.0 µs of 102.0 µs)"));
  EXPECT_TRUE(contains(verdict, "Spread: stock 4.0 µs, custom 3.0 µs."));
  EXPECT_TRUE(contains(verdict, "The difference exceeds the spread."));
}

void should_state_a_slower_candidate_as_slower() {
  const std::string verdict = describe_verdict(
      "stock", "custom", compare_paired_samples({90, 93, 91}, {100, 104, 102}));
  EXPECT_TRUE(contains(verdict, "custom is 12.1% slower than stock"));
}

void should_say_a_difference_inside_the_spread_is_not_a_result() {
  const std::string verdict = describe_verdict(
      "stock", "custom", compare_paired_samples({100, 110, 120}, {95, 105, 115}));
  EXPECT_TRUE(contains(verdict, "The 5.0 µs difference between custom and stock is inside"
                                " the spread: not a result."));
}

void should_render_all_three_metrics_with_ranges_and_rounds() {
  PairedMeasurement paired_measurement;
  report_fixture::populate_pair(paired_measurement);
  const std::string section_html =
      render_verdict_section("stock", "custom", compare_pair(paired_measurement),
                             paired_measurement.output_agreement);
  EXPECT_TRUE(contains(section_html, "class=\"verdict within-spread\""));
  for (const char* metric_name : {"Job duration", "GPU execution", "Host submission"}) {
    EXPECT_TRUE(contains(section_html, metric_name));
  }
  EXPECT_TRUE(contains(section_html, "faster, beyond the spread"));
  EXPECT_TRUE(contains(section_html, "inside the spread: not a result"));
  EXPECT_TRUE(contains(section_html, "3 of 3"));
  EXPECT_TRUE(contains(section_html, "largest element-wise difference is 0.25"));
}

// --- edge / boundary ---

void should_describe_equal_variants_without_dividing_by_zero() {
  const std::string verdict =
      describe_verdict("stock", "custom", compare_paired_samples({0, 0}, {0, 0}));
  EXPECT_TRUE(contains(verdict, "The 0.0 µs difference"));
}

// --- negative: names are the caller's text, never markup ---

void should_escape_variant_names() {
  const std::string section_html = render_verdict_section(
      "<b>stock</b>", "custom & co",
      {compare_paired_samples({10}, {8}), compare_paired_samples({10}, {8}),
       compare_paired_samples({10}, {8})},
      {1, 0.0});
  EXPECT_TRUE(!contains(section_html, "<b>stock</b>"));
  EXPECT_TRUE(contains(section_html, "&lt;b&gt;stock&lt;/b&gt;"));
  EXPECT_TRUE(contains(section_html, "custom &amp; co"));
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_state_which_is_faster_by_how_much_and_against_the_spread),
      TEST_CASE(should_state_a_slower_candidate_as_slower),
      TEST_CASE(should_say_a_difference_inside_the_spread_is_not_a_result),
      TEST_CASE(should_render_all_three_metrics_with_ranges_and_rounds),
      TEST_CASE(should_describe_equal_variants_without_dividing_by_zero),
      TEST_CASE(should_escape_variant_names),
  });
}
