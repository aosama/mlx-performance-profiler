#include "sample_statistics.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace mlx_perf {

namespace {

// The `fraction` percentile of non-empty sorted samples, interpolating
// linearly at position fraction·(n − 1).
double interpolate_percentile(const std::vector<double>& sorted_samples, double fraction) {
  const double position = fraction * static_cast<double>(sorted_samples.size() - 1);
  const std::size_t lower_index = static_cast<std::size_t>(position);
  const std::size_t upper_index = std::min(lower_index + 1, sorted_samples.size() - 1);
  const double upper_weight = position - static_cast<double>(lower_index);
  return sorted_samples[lower_index] * (1.0 - upper_weight) +
         sorted_samples[upper_index] * upper_weight;
}

} // namespace

SampleSummary summarize_samples(const std::vector<double>& samples) {
  if (samples.empty()) {
    throw std::invalid_argument("summarize_samples: no samples");
  }
  std::vector<double> sorted_samples = samples;
  std::sort(sorted_samples.begin(), sorted_samples.end());
  SampleSummary sample_summary;
  sample_summary.median = interpolate_percentile(sorted_samples, 0.5);
  sample_summary.lower_quartile = interpolate_percentile(sorted_samples, 0.25);
  sample_summary.upper_quartile = interpolate_percentile(sorted_samples, 0.75);
  sample_summary.minimum = sorted_samples.front();
  sample_summary.maximum = sorted_samples.back();
  sample_summary.sample_count = static_cast<int>(sorted_samples.size());
  return sample_summary;
}

std::string_view comparison_verdict_name(ComparisonVerdict verdict) {
  switch (verdict) {
    case ComparisonVerdict::candidate_faster: return "candidate_faster";
    case ComparisonVerdict::candidate_slower: return "candidate_slower";
    case ComparisonVerdict::within_spread: break;
  }
  return "within_spread";
}

PairedComparison compare_paired_samples(const std::vector<double>& baseline_samples,
                                        const std::vector<double>& candidate_samples) {
  if (baseline_samples.size() != candidate_samples.size()) {
    throw std::invalid_argument("compare_paired_samples: " +
                                std::to_string(baseline_samples.size()) + " baseline vs " +
                                std::to_string(candidate_samples.size()) +
                                " candidate samples; rounds must pair");
  }
  PairedComparison paired_comparison;
  paired_comparison.baseline = summarize_samples(baseline_samples);
  paired_comparison.candidate = summarize_samples(candidate_samples);
  paired_comparison.round_count = static_cast<int>(baseline_samples.size());
  std::vector<double> round_differences;
  for (std::size_t round_index = 0; round_index < baseline_samples.size(); ++round_index) {
    round_differences.push_back(candidate_samples[round_index] - baseline_samples[round_index]);
    paired_comparison.rounds_candidate_faster += round_differences.back() < 0.0 ? 1 : 0;
  }
  paired_comparison.round_difference = summarize_samples(round_differences);
  paired_comparison.median_difference = paired_comparison.round_difference.median;
  paired_comparison.relative_difference = paired_comparison.baseline.median == 0.0
      ? 0.0
      : paired_comparison.median_difference / paired_comparison.baseline.median;
  if (paired_comparison.round_difference.upper_quartile < 0.0) {
    paired_comparison.verdict = ComparisonVerdict::candidate_faster;
  } else if (paired_comparison.round_difference.lower_quartile > 0.0) {
    paired_comparison.verdict = ComparisonVerdict::candidate_slower;
  }
  return paired_comparison;
}

} // namespace mlx_perf
