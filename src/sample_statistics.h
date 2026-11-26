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
  double median_difference = 0.0;    // candidate − baseline; negative = faster
  double relative_difference = 0.0;  // median_difference / baseline median; 0 when that is 0
  ComparisonVerdict verdict = ComparisonVerdict::within_spread;
  int rounds_candidate_faster = 0;   // rounds where the candidate's sample was lower
  int round_count = 0;
};

// Lower is better. The candidate is faster only when the middle half of its
// samples lies entirely below the middle half of the baseline's (its upper
// quartile is below the baseline's lower quartile), slower in the mirror
// case; otherwise the difference is inside the spread and not a result.
//
// Why the middle half and not the full range: on a machine doing other work
// one round in five or seven can be disturbed (measured: a 34–39 µs kernel
// reading 578 µs once). With full ranges, that single round vetoes a 6×
// difference; the middle half ignores it while still refusing any
// difference the bulk of the rounds do not support. Samples are paired by
// round (index), which is what rounds_candidate_faster counts; both vectors
// must have the same, non-zero length or std::invalid_argument is thrown.
PairedComparison compare_paired_samples(const std::vector<double>& baseline_samples,
                                        const std::vector<double>& candidate_samples);

} // namespace mlx_perf
