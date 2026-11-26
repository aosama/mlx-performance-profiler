#include "sample_statistics.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace mlx_perf {

SampleSummary summarize_samples(const std::vector<double>& samples) {
  if (samples.empty()) {
    throw std::invalid_argument("summarize_samples: no samples");
  }
  std::vector<double> sorted_samples = samples;
  std::sort(sorted_samples.begin(), sorted_samples.end());
  const std::size_t middle_index = sorted_samples.size() / 2;
  SampleSummary sample_summary;
  sample_summary.median = sorted_samples.size() % 2 == 1
      ? sorted_samples[middle_index]
      : (sorted_samples[middle_index - 1] + sorted_samples[middle_index]) / 2.0;
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
  paired_comparison.median_difference =
      paired_comparison.candidate.median - paired_comparison.baseline.median;
  paired_comparison.relative_difference = paired_comparison.baseline.median == 0.0
      ? 0.0
      : paired_comparison.median_difference / paired_comparison.baseline.median;
  for (std::size_t round_index = 0; round_index < baseline_samples.size(); ++round_index) {
    paired_comparison.rounds_candidate_faster +=
        candidate_samples[round_index] < baseline_samples[round_index] ? 1 : 0;
  }
  if (paired_comparison.candidate.maximum < paired_comparison.baseline.minimum) {
    paired_comparison.verdict = ComparisonVerdict::candidate_faster;
  } else if (paired_comparison.candidate.minimum > paired_comparison.baseline.maximum) {
    paired_comparison.verdict = ComparisonVerdict::candidate_slower;
  }
  return paired_comparison;
}

} // namespace mlx_perf
