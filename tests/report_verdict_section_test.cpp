// Component tests for report_verdict_section: the verdict sentences and the
// metric table, from synthetic comparisons (no GPU).

#include "report_fixture.h"
#include "report_verdict_section.h"
#include "test_harness.h"

#include <cmath>
#include <string>
#include <vector>

namespace {

using namespace mlx_perf;

bool contains(const std::string& text, const std::string& needle) {
  return text.find(needle) != std::string::npos;
}

const std::vector<double> TWELVE_BASELINE_ROUNDS = {100, 104, 102, 101, 103, 99,
                                                   105, 100, 102, 101, 104, 103};

std::vector<double> shifted(const std::vector<double>& round_shifts) {
  std::vector<double> candidate_rounds;
  for (std::size_t round_index = 0; round_index < round_shifts.size(); ++round_index) {
    candidate_rounds.push_back(TWELVE_BASELINE_ROUNDS[round_index] + round_shifts[round_index]);
  }
  return candidate_rounds;
}

// --- positive ---

void should_state_which_is_faster_by_how_much_in_how_many_rounds_and_the_interval() {
  const PairedComparison paired_comparison = compare_paired_samples(
      TWELVE_BASELINE_ROUNDS, shifted({-12, -10, -11, -9, -13, -10, -8, -11, -12, -10, -9, -14}));
  const std::string verdict = describe_verdict("stock", "custom", paired_comparison);
  EXPECT_TRUE(contains(verdict, "custom is 10.3% faster than stock"));
  EXPECT_TRUE(contains(verdict, "(10.5 µs of 102.0 µs, faster in 12 of 12 rounds)."));
  EXPECT_TRUE(contains(verdict, "96% interval for the per-round difference: −12.0 µs to −9.0 µs."));
  EXPECT_TRUE(contains(verdict, "The difference exceeds the spread."));
}

void should_state_a_slower_candidate_with_the_rounds_it_lost() {
  const std::string verdict = describe_verdict(
      "stock", "custom", compare_paired_samples(TWELVE_BASELINE_ROUNDS,
                                                shifted(std::vector<double>(12, 7.0))));
  EXPECT_TRUE(contains(verdict, "custom is 6.9% slower than stock"));
  EXPECT_TRUE(contains(verdict, "slower in 12 of 12 rounds"));
}

void should_say_an_interval_including_zero_is_not_a_result() {
  const std::string verdict = describe_verdict(
      "stock", "custom",
      compare_paired_samples(TWELVE_BASELINE_ROUNDS,
                             shifted({5, -10, 5, -3, 4, -6, 2, -1, 3, -2, 1, -4})));
  EXPECT_TRUE(contains(verdict, "difference between custom and stock is not a result: the 96%"
                                " interval for the per-round difference"));
  EXPECT_TRUE(contains(verdict, "includes zero."));
}

void should_render_all_three_metrics_with_intervals_and_rounds() {
  PairedMeasurement paired_measurement;
  report_fixture::populate_pair(paired_measurement);
  const std::string section_html =
      render_verdict_section("stock", "custom", compare_pair(paired_measurement),
                             paired_measurement.output_agreement);
  EXPECT_TRUE(contains(section_html, "class=\"verdict within-spread\""));
  for (const char* metric_name : {"Job duration", "GPU span", "Host submission"}) {
    EXPECT_TRUE(contains(section_html, metric_name));
  }
  EXPECT_TRUE(contains(section_html, "faster, beyond the spread"));
  EXPECT_TRUE(contains(section_html, "inside the spread: not a result"));
  EXPECT_TRUE(contains(section_html, "12 of 12"));
  EXPECT_TRUE(contains(section_html, "(96%)"));
  EXPECT_TRUE(contains(section_html, "largest element-wise difference is 0.25"));
}

// --- edge / boundary ---

void should_say_too_few_rounds_rather_than_judge() {
  const std::string verdict =
      describe_verdict("stock", "custom", compare_paired_samples({10, 10, 10}, {8, 8, 8}));
  EXPECT_TRUE(contains(verdict, "is not a result: 3 rounds are too few to judge."));
}

void should_describe_equal_variants_without_dividing_by_zero() {
  const std::vector<double> zero_rounds(12, 0.0);
  const std::string verdict =
      describe_verdict("stock", "custom", compare_paired_samples(zero_rounds, zero_rounds));
  EXPECT_TRUE(contains(verdict, "The 0.0 µs difference"));
}

void should_say_when_output_agreement_cannot_be_measured() {
  const std::string section_html = render_verdict_section(
      "stock", "custom",
      {compare_paired_samples({10}, {8}), compare_paired_samples({10}, {8}),
       compare_paired_samples({10}, {8})},
      {1, std::nan("")});
  EXPECT_TRUE(contains(section_html, "not finite (NaN or infinity)"));
  EXPECT_TRUE(!contains(section_html, "largest element-wise difference is"));
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
      TEST_CASE(should_state_which_is_faster_by_how_much_in_how_many_rounds_and_the_interval),
      TEST_CASE(should_state_a_slower_candidate_with_the_rounds_it_lost),
      TEST_CASE(should_say_an_interval_including_zero_is_not_a_result),
      TEST_CASE(should_render_all_three_metrics_with_intervals_and_rounds),
      TEST_CASE(should_say_too_few_rounds_rather_than_judge),
      TEST_CASE(should_describe_equal_variants_without_dividing_by_zero),
      TEST_CASE(should_say_when_output_agreement_cannot_be_measured),
      TEST_CASE(should_escape_variant_names),
  });
}
