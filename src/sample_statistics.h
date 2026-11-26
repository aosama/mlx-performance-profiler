#pragma once

// Sample statistics: one concern only — summarise repeated measurements and
// decide whether two variants differ by more than their run-to-run spread.
// Pure functions over numbers, free of MLX, so the rule is unit-tested
// without a GPU.

#include <string_view>
#include <vector>

namespace mlx_perf {

struct SampleSummary {
  double median = 0.0;
  double minimum = 0.0;
  double maximum = 0.0;
  // 25th and 75th percentiles: the middle half of the samples.
  double lower_quartile = 0.0;
  double upper_quartile = 0.0;
  int sample_count = 0;
  double interquartile_range() const { return upper_quartile - lower_quartile; }
};

// Percentiles interpolate linearly between the two nearest sorted samples
// (position p·(n − 1), numpy's default), so an even count's median is the
// mean of the two middle samples. Throws std::invalid_argument on an empty
// sample set: a summary of nothing would read as a measured zero.
SampleSummary summarize_samples(const std::vector<double>& samples);

enum class ComparisonVerdict { candidate_faster, within_spread, candidate_slower };

std::string_view comparison_verdict_name(ComparisonVerdict verdict);

// The interval is valid at this coverage or better; below it, too few rounds
// were measured to judge, and the verdict is always "within spread".
inline constexpr double MINIMUM_INTERVAL_COVERAGE = 0.95;

struct PairedComparison {
  SampleSummary baseline;
  SampleSummary candidate;
  // candidate − baseline, round by round: what the verdict is judged on.
  SampleSummary round_difference;
  double median_difference = 0.0;    // round_difference.median; negative = faster
  double relative_difference = 0.0;  // median_difference / baseline median; 0 when that is 0
  // A distribution-free interval for the median per-round difference: the
  // k-th smallest and k-th largest differences, with the largest k whose
  // coverage under the sign test is at least MINIMUM_INTERVAL_COVERAGE (12
  // rounds: the 3rd and 10th smallest, 96.1%). With too few rounds no k
  // qualifies; the interval is then the full range at its actual coverage.
  double difference_interval_low = 0.0;
  double difference_interval_high = 0.0;
  double difference_interval_coverage = 0.0;
  // Exact two-sided sign test over the non-zero differences; 1 when every
  // difference is zero. Reported for readers who want it; the verdict
  // follows the interval.
  double sign_test_p_value = 1.0;
  ComparisonVerdict verdict = ComparisonVerdict::within_spread;
  int rounds_candidate_faster = 0;   // rounds where the candidate's sample was lower
  int round_count = 0;
};

// Lower is better. The candidate is faster only when the whole interval for
// the median per-round difference lies below zero, slower when it lies
// above, and never when the interval's coverage is below
// MINIMUM_INTERVAL_COVERAGE. With 12 rounds that means winning at least 10
// of 12; two identical variants are called different in at most 3.9% of
// comparisons (two-sided).
//
// Why this rule, after two that failed on measurements:
//  - Per-round differences, not each variant's own spread: on a decoder step
//    job wall stepped up ~1 ms mid-run for BOTH variants; differences cancel
//    what a round's two runs share.
//  - Signs, not magnitudes: one disturbed round (578 µs for a 37 µs kernel)
//    counts as one round lost, however large it is.
//  - An exact interval, not quartiles: "the middle half of the differences
//    is below zero" fired for identical variants in 22–38% of comparisons.
// Both vectors must have the same, non-zero length or std::invalid_argument
// is thrown.
PairedComparison compare_paired_samples(const std::vector<double>& baseline_samples,
                                        const std::vector<double>& candidate_samples);

} // namespace mlx_perf
