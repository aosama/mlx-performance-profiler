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

struct PairedComparison {
  SampleSummary baseline;
  SampleSummary candidate;
  // candidate − baseline, round by round: what the verdict is judged on.
  SampleSummary round_difference;
  double median_difference = 0.0;    // round_difference.median; negative = faster
  double relative_difference = 0.0;  // median_difference / baseline median; 0 when that is 0
  ComparisonVerdict verdict = ComparisonVerdict::within_spread;
  int rounds_candidate_faster = 0;   // rounds where the candidate's sample was lower
  int round_count = 0;
};

// Lower is better. The verdict is judged on the per-round differences
// (candidate − baseline, paired by index): the candidate is faster only when
// the middle half of those differences (lower to upper quartile) lies
// entirely below zero, slower when it lies entirely above, otherwise the
// difference is inside the spread and not a result.
//
// Why differences, and why their middle half (both measured on the GPU):
//  - Drift is shared within a round: on a decoder step, job wall stepped up
//    by ~1 ms part-way through a run for BOTH variants. Each variant's own
//    spread then swallowed a difference the candidate won in 7 of 7 rounds;
//    the per-round difference cancels what both rounds share.
//  - One round can be disturbed (a 34–39 µs kernel read 578 µs once). The
//    middle half ignores a single outlier while still refusing any
//    difference the bulk of the rounds do not support.
// Both vectors must have the same, non-zero length or std::invalid_argument
// is thrown.
PairedComparison compare_paired_samples(const std::vector<double>& baseline_samples,
                                        const std::vector<double>& candidate_samples);

} // namespace mlx_perf
