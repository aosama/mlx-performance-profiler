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
  int sample_count = 0;
  double spread() const { return maximum - minimum; }
};

// Throws std::invalid_argument on an empty sample set: a summary of nothing
// would read as a measured zero. An even count takes the mean of the two
// middle samples.
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

// Lower is better. The verdict is strict on purpose, so it can be stated in
// one sentence on the page: the candidate is faster only when EVERY one of
// its samples beats EVERY baseline sample (the two ranges do not overlap),
// slower in the mirror case, otherwise the difference is inside the spread
// and not a result. Samples are paired by round (index), which is what
// rounds_candidate_faster counts; both vectors must have the same, non-zero
// length or std::invalid_argument is thrown.
PairedComparison compare_paired_samples(const std::vector<double>& baseline_samples,
                                        const std::vector<double>& candidate_samples);

} // namespace mlx_perf
