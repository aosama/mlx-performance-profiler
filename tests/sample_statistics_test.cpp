// Unit tests for sample_statistics: medians, quartiles and the verdict
// "the middle halves do not overlap", on hand-picked numbers (no GPU).

#include "sample_statistics.h"
#include "test_harness.h"

#include <stdexcept>
#include <vector>

namespace {

using mlx_perf::ComparisonVerdict;
using mlx_perf::compare_paired_samples;
using mlx_perf::summarize_samples;

bool throws_invalid_argument(const std::vector<double>& baseline_samples,
                             const std::vector<double>& candidate_samples) {
  try {
    compare_paired_samples(baseline_samples, candidate_samples);
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

// --- positive ---

void should_summarize_median_and_range_of_an_odd_count() {
  const auto sample_summary = summarize_samples({30, 10, 20, 50, 40});
  EXPECT_TRUE(sample_summary.median == 30);
  EXPECT_TRUE(sample_summary.minimum == 10);
  EXPECT_TRUE(sample_summary.maximum == 50);
  EXPECT_TRUE(sample_summary.lower_quartile == 20);
  EXPECT_TRUE(sample_summary.upper_quartile == 40);
  EXPECT_TRUE(sample_summary.interquartile_range() == 20);
  EXPECT_TRUE(sample_summary.sample_count == 5);
}

void should_call_a_candidate_faster_when_its_middle_half_lies_below_the_baselines() {
  const auto paired_comparison = compare_paired_samples({100, 104, 102}, {90, 93, 91});
  EXPECT_TRUE(paired_comparison.verdict == ComparisonVerdict::candidate_faster);
  EXPECT_TRUE(paired_comparison.median_difference == -11);
  EXPECT_NEAR(paired_comparison.relative_difference, -11.0 / 102.0, 1e-12);
  EXPECT_TRUE(paired_comparison.rounds_candidate_faster == 3);
  EXPECT_TRUE(paired_comparison.round_count == 3);
}

// Measured on the GPU: a fused kernel took 34–39 µs except once (578 µs).
// Full ranges would overlap; the middle halves do not.
void should_not_let_one_disturbed_round_veto_a_clear_difference() {
  const auto paired_comparison = compare_paired_samples({247.5, 221.5, 234.5, 229.5, 220.0},
                                                        {38.6, 38.4, 578.3, 37.6, 33.9});
  EXPECT_TRUE(paired_comparison.candidate.maximum > paired_comparison.baseline.minimum);
  EXPECT_TRUE(paired_comparison.verdict == ComparisonVerdict::candidate_faster);
  EXPECT_TRUE(paired_comparison.rounds_candidate_faster == 4);
}

void should_call_a_candidate_slower_in_the_mirror_case() {
  const auto paired_comparison = compare_paired_samples({90, 93, 91}, {100, 104, 102});
  EXPECT_TRUE(paired_comparison.verdict == ComparisonVerdict::candidate_slower);
  EXPECT_TRUE(paired_comparison.rounds_candidate_faster == 0);
}

void should_call_overlapping_middle_halves_within_the_spread() {
  // Medians differ by 5, but the ranges overlap: not a result.
  const auto paired_comparison = compare_paired_samples({100, 110, 120}, {95, 105, 115});
  EXPECT_TRUE(paired_comparison.verdict == ComparisonVerdict::within_spread);
  EXPECT_TRUE(paired_comparison.median_difference == -5);
  EXPECT_TRUE(paired_comparison.rounds_candidate_faster == 3);
}

void should_name_every_verdict() {
  EXPECT_TRUE(mlx_perf::comparison_verdict_name(ComparisonVerdict::candidate_faster) ==
              "candidate_faster");
  EXPECT_TRUE(mlx_perf::comparison_verdict_name(ComparisonVerdict::within_spread) ==
              "within_spread");
  EXPECT_TRUE(mlx_perf::comparison_verdict_name(ComparisonVerdict::candidate_slower) ==
              "candidate_slower");
}

// --- edge / boundary ---

void should_interpolate_the_median_and_quartiles_of_an_even_count() {
  const auto sample_summary = summarize_samples({4, 1, 3, 2});
  EXPECT_TRUE(sample_summary.median == 2.5);
  EXPECT_TRUE(sample_summary.lower_quartile == 1.75);
  EXPECT_TRUE(sample_summary.upper_quartile == 3.25);
}

void should_treat_middle_halves_touching_at_one_value_as_overlapping() {
  // Candidate upper quartile (102.5) equals the baseline's lower quartile:
  // not strictly below, so not a result.
  const auto paired_comparison = compare_paired_samples({100, 110}, {95, 105});
  EXPECT_TRUE(paired_comparison.candidate.upper_quartile ==
              paired_comparison.baseline.lower_quartile);
  EXPECT_TRUE(paired_comparison.verdict == ComparisonVerdict::within_spread);
}

void should_handle_a_single_round_and_identical_samples() {
  const auto single_round = compare_paired_samples({10}, {8});
  EXPECT_TRUE(single_round.verdict == ComparisonVerdict::candidate_faster);
  EXPECT_TRUE(single_round.baseline.interquartile_range() == 0);
  const auto identical_samples = compare_paired_samples({7, 7, 7}, {7, 7, 7});
  EXPECT_TRUE(identical_samples.verdict == ComparisonVerdict::within_spread);
  EXPECT_TRUE(identical_samples.median_difference == 0);
}

void should_report_no_relative_difference_against_a_zero_baseline() {
  const auto paired_comparison = compare_paired_samples({0, 0}, {1, 2});
  EXPECT_TRUE(paired_comparison.relative_difference == 0);
  EXPECT_TRUE(paired_comparison.verdict == ComparisonVerdict::candidate_slower);
}

// --- negative ---

void should_reject_an_empty_sample_set() {
  bool has_thrown = false;
  try {
    summarize_samples({});
  } catch (const std::invalid_argument&) {
    has_thrown = true;
  }
  EXPECT_TRUE(has_thrown);
  EXPECT_TRUE(throws_invalid_argument({}, {}));
}

void should_reject_rounds_that_do_not_pair() {
  EXPECT_TRUE(throws_invalid_argument({1, 2, 3}, {1, 2}));
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_summarize_median_and_range_of_an_odd_count),
      TEST_CASE(should_call_a_candidate_faster_when_its_middle_half_lies_below_the_baselines),
      TEST_CASE(should_not_let_one_disturbed_round_veto_a_clear_difference),
      TEST_CASE(should_call_a_candidate_slower_in_the_mirror_case),
      TEST_CASE(should_call_overlapping_middle_halves_within_the_spread),
      TEST_CASE(should_name_every_verdict),
      TEST_CASE(should_interpolate_the_median_and_quartiles_of_an_even_count),
      TEST_CASE(should_treat_middle_halves_touching_at_one_value_as_overlapping),
      TEST_CASE(should_handle_a_single_round_and_identical_samples),
      TEST_CASE(should_report_no_relative_difference_against_a_zero_baseline),
      TEST_CASE(should_reject_an_empty_sample_set),
      TEST_CASE(should_reject_rounds_that_do_not_pair),
  });
}
