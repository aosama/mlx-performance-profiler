// Unit tests for sample_statistics: medians and quartiles, the sign-test
// interval for the median per-round difference, and the verdict, on
// hand-picked, measured and simulated numbers (no GPU).

#include "sample_statistics.h"
#include "test_harness.h"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using mlx_perf::ComparisonVerdict;
using mlx_perf::compare_paired_samples;
using mlx_perf::summarize_samples;

constexpr double PROBABILITY_TOLERANCE = 1e-12;

bool throws_invalid_argument(const std::vector<double>& baseline_samples,
                             const std::vector<double>& candidate_samples) {
  try {
    compare_paired_samples(baseline_samples, candidate_samples);
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

// The baseline's rounds, each moved by its entry in `round_shifts`.
std::vector<double> shifted(const std::vector<double>& baseline_samples,
                            const std::vector<double>& round_shifts) {
  std::vector<double> candidate_samples;
  for (std::size_t round_index = 0; round_index < baseline_samples.size(); ++round_index) {
    candidate_samples.push_back(baseline_samples[round_index] + round_shifts[round_index]);
  }
  return candidate_samples;
}

const std::vector<double> TWELVE_BASELINE_ROUNDS = {100, 104, 102, 101, 103, 99,
                                                   105, 100, 102, 101, 104, 103};

// --- positive ---

void should_summarize_median_and_quartiles_of_an_odd_count() {
  const auto sample_summary = summarize_samples({30, 10, 20, 50, 40});
  EXPECT_TRUE(sample_summary.median == 30);
  EXPECT_TRUE(sample_summary.minimum == 10);
  EXPECT_TRUE(sample_summary.maximum == 50);
  EXPECT_TRUE(sample_summary.lower_quartile == 20);
  EXPECT_TRUE(sample_summary.upper_quartile == 40);
  EXPECT_TRUE(sample_summary.sample_count == 5);
}

// 12 of 12 rounds faster: the interval is the 3rd and 10th smallest
// difference, at 1 − 2·79/4096 = 96.14% coverage; p = 2/4096.
void should_call_a_candidate_faster_when_the_whole_interval_is_below_zero() {
  const std::vector<double> round_shifts = {-12, -10, -11, -9, -13, -10,
                                            -8,  -11, -12, -10, -9, -14};
  const auto paired_comparison = compare_paired_samples(
      TWELVE_BASELINE_ROUNDS, shifted(TWELVE_BASELINE_ROUNDS, round_shifts));
  EXPECT_TRUE(paired_comparison.verdict == ComparisonVerdict::candidate_faster);
  EXPECT_TRUE(paired_comparison.difference_interval_low == -12);  // 3rd smallest
  EXPECT_TRUE(paired_comparison.difference_interval_high == -9);  // 10th smallest
  EXPECT_NEAR(paired_comparison.difference_interval_coverage, 1.0 - 2.0 * 79.0 / 4096.0,
              PROBABILITY_TOLERANCE);
  EXPECT_NEAR(paired_comparison.sign_test_p_value, 2.0 / 4096.0, PROBABILITY_TOLERANCE);
  EXPECT_TRUE(paired_comparison.rounds_candidate_faster == 12);
  EXPECT_TRUE(paired_comparison.median_difference == -10.5);
}

// Measured shape: a 37 µs kernel against a 230 µs one, with two disturbed
// rounds where the fast kernel read 578 µs. 10 of 12 still decide it.
void should_not_let_disturbed_rounds_veto_a_clear_difference() {
  const std::vector<double> baseline_rounds = {243, 234, 249, 249, 243, 232,
                                               235, 235, 222, 236, 238, 230};
  const std::vector<double> candidate_rounds = {38, 37, 578, 36, 33, 38,
                                                33, 36, 37, 578, 38, 37};
  const auto paired_comparison = compare_paired_samples(baseline_rounds, candidate_rounds);
  EXPECT_TRUE(paired_comparison.rounds_candidate_faster == 10);
  EXPECT_TRUE(paired_comparison.verdict == ComparisonVerdict::candidate_faster);
  EXPECT_NEAR(paired_comparison.sign_test_p_value, 2.0 * 79.0 / 4096.0, PROBABILITY_TOLERANCE);
}

// Measured shape: job wall stepped up ~1 ms part-way through the run for
// BOTH variants. Each variant's own spread swallows the difference; the
// per-round differences do not.
void should_see_through_drift_both_variants_share() {
  const std::vector<double> baseline_rounds = {3150, 3140, 3120, 3250, 3180, 3160,
                                               4380, 4270, 4290, 4310, 4260, 4300};
  const std::vector<double> round_shifts = {-150, -20,  -140, -300, -90,  -160,
                                            -410, -340, -330, -280, -250, -300};
  const auto paired_comparison =
      compare_paired_samples(baseline_rounds, shifted(baseline_rounds, round_shifts));
  EXPECT_TRUE(paired_comparison.candidate.upper_quartile >
              paired_comparison.baseline.lower_quartile);
  EXPECT_TRUE(paired_comparison.verdict == ComparisonVerdict::candidate_faster);
}

void should_call_a_candidate_slower_in_the_mirror_case() {
  const std::vector<double> round_shifts(12, 7.0);
  const auto paired_comparison = compare_paired_samples(
      TWELVE_BASELINE_ROUNDS, shifted(TWELVE_BASELINE_ROUNDS, round_shifts));
  EXPECT_TRUE(paired_comparison.verdict == ComparisonVerdict::candidate_slower);
  EXPECT_TRUE(paired_comparison.rounds_candidate_faster == 0);
}

void should_name_every_verdict() {
  EXPECT_TRUE(mlx_perf::comparison_verdict_name(ComparisonVerdict::candidate_faster) ==
              "candidate_faster");
  EXPECT_TRUE(mlx_perf::comparison_verdict_name(ComparisonVerdict::within_spread) ==
              "within_spread");
  EXPECT_TRUE(mlx_perf::comparison_verdict_name(ComparisonVerdict::candidate_slower) ==
              "candidate_slower");
}

// The guarantee behind the rule: two identical variants with independent
// noise are called different in at most ~3.9% of comparisons. Seeded, so
// the test is deterministic; 4000 trials put the rate's standard error near
// 0.3%, well inside the 5% bound asserted.
void should_rarely_call_identical_variants_different() {
  std::mt19937 noise_generator(20261126);
  std::normal_distribution<double> round_noise(0.0, 1.0);
  constexpr int TRIAL_COUNT = 4000;
  int false_result_count = 0;
  for (int trial_index = 0; trial_index < TRIAL_COUNT; ++trial_index) {
    std::vector<double> baseline_rounds;
    std::vector<double> candidate_rounds;
    for (int round_index = 0; round_index < 12; ++round_index) {
      baseline_rounds.push_back(100.0 + round_noise(noise_generator));
      candidate_rounds.push_back(100.0 + round_noise(noise_generator));
    }
    const bool is_called_different =
        compare_paired_samples(baseline_rounds, candidate_rounds).verdict !=
        ComparisonVerdict::within_spread;
    false_result_count += is_called_different ? 1 : 0;
  }
  EXPECT_TRUE(static_cast<double>(false_result_count) / TRIAL_COUNT <= 0.05);
}

// --- edge / boundary ---

void should_interpolate_the_median_and_quartiles_of_an_even_count() {
  const auto sample_summary = summarize_samples({4, 1, 3, 2});
  EXPECT_TRUE(sample_summary.median == 2.5);
  EXPECT_TRUE(sample_summary.lower_quartile == 1.75);
  EXPECT_TRUE(sample_summary.upper_quartile == 3.25);
}

// 9 of 12 is p = 2·299/4096 ≈ 0.146: the interval reaches back over zero.
void should_call_nine_of_twelve_rounds_not_a_result() {
  const std::vector<double> round_shifts = {-12, -10, 4, -9, -13, 3, -8, -11, 6, -10, -9, -14};
  const auto paired_comparison = compare_paired_samples(
      TWELVE_BASELINE_ROUNDS, shifted(TWELVE_BASELINE_ROUNDS, round_shifts));
  EXPECT_TRUE(paired_comparison.rounds_candidate_faster == 9);
  EXPECT_TRUE(paired_comparison.difference_interval_high > 0);
  EXPECT_TRUE(paired_comparison.verdict == ComparisonVerdict::within_spread);
}

// Five rounds cannot reach 95% coverage (the best is 1 − 2/32 = 93.75%), so
// even five of five is not a result.
void should_never_judge_a_comparison_with_too_few_rounds() {
  const auto paired_comparison = compare_paired_samples({10, 10, 10, 10, 10}, {8, 8, 8, 8, 8});
  EXPECT_TRUE(paired_comparison.difference_interval_coverage < 0.95);
  EXPECT_TRUE(paired_comparison.verdict == ComparisonVerdict::within_spread);
  EXPECT_TRUE(compare_paired_samples({10}, {8}).verdict == ComparisonVerdict::within_spread);
}

void should_treat_identical_samples_as_no_difference_with_p_of_one() {
  const auto paired_comparison =
      compare_paired_samples(TWELVE_BASELINE_ROUNDS, TWELVE_BASELINE_ROUNDS);
  EXPECT_TRUE(paired_comparison.verdict == ComparisonVerdict::within_spread);
  EXPECT_TRUE(paired_comparison.median_difference == 0);
  EXPECT_TRUE(paired_comparison.sign_test_p_value == 1.0);
}

void should_report_no_relative_difference_against_a_zero_baseline() {
  const std::vector<double> zero_baseline(12, 0.0);
  const auto paired_comparison =
      compare_paired_samples(zero_baseline, std::vector<double>(12, 2.0));
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
      TEST_CASE(should_summarize_median_and_quartiles_of_an_odd_count),
      TEST_CASE(should_call_a_candidate_faster_when_the_whole_interval_is_below_zero),
      TEST_CASE(should_not_let_disturbed_rounds_veto_a_clear_difference),
      TEST_CASE(should_see_through_drift_both_variants_share),
      TEST_CASE(should_call_a_candidate_slower_in_the_mirror_case),
      TEST_CASE(should_name_every_verdict),
      TEST_CASE(should_rarely_call_identical_variants_different),
      TEST_CASE(should_interpolate_the_median_and_quartiles_of_an_even_count),
      TEST_CASE(should_call_nine_of_twelve_rounds_not_a_result),
      TEST_CASE(should_never_judge_a_comparison_with_too_few_rounds),
      TEST_CASE(should_treat_identical_samples_as_no_difference_with_p_of_one),
      TEST_CASE(should_report_no_relative_difference_against_a_zero_baseline),
      TEST_CASE(should_reject_an_empty_sample_set),
      TEST_CASE(should_reject_rounds_that_do_not_pair),
  });
}
