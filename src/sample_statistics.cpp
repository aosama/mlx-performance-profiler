#include "sample_statistics.h"

#include <algorithm>
#include <cmath>
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

// P(X ≤ success_limit) for X ~ Binomial(trial_count, 1/2): the sign test's
// null distribution. Exact enough in doubles for the round counts used here.
double binomial_half_cumulative(int trial_count, int success_limit) {
  double cumulative_probability = 0.0;
  double combinations = 1.0;  // C(trial_count, successes), built up incrementally
  for (int successes = 0; successes <= success_limit && successes <= trial_count; ++successes) {
    if (successes > 0) {
      combinations = combinations * (trial_count - successes + 1) / successes;
    }
    cumulative_probability += combinations;
  }
  return cumulative_probability / std::pow(2.0, trial_count);
}

struct OrderStatisticInterval {
  double low = 0.0;
  double high = 0.0;
  double coverage = 0.0;
};

// [x(k), x(n−k+1)] covers the median with probability 1 − 2·P(X ≤ k−1).
// Takes the largest k (narrowest interval) that still reaches the minimum
// coverage, or the full range when none does.
OrderStatisticInterval interval_for_median(const std::vector<double>& sorted_differences) {
  const int round_count = static_cast<int>(sorted_differences.size());
  OrderStatisticInterval median_interval{sorted_differences.front(), sorted_differences.back(),
                                         1.0 - 2.0 * binomial_half_cumulative(round_count, 0)};
  for (int order_index = 2; order_index <= (round_count + 1) / 2; ++order_index) {
    const double coverage = 1.0 - 2.0 * binomial_half_cumulative(round_count, order_index - 1);
    if (coverage < MINIMUM_INTERVAL_COVERAGE) {
      break;
    }
    median_interval = {sorted_differences[order_index - 1],
                       sorted_differences[round_count - order_index], coverage};
  }
  return median_interval;
}

double sign_test_p_value(const std::vector<double>& round_differences) {
  int negative_count = 0;
  int positive_count = 0;
  for (const double round_difference : round_differences) {
    negative_count += round_difference < 0.0 ? 1 : 0;
    positive_count += round_difference > 0.0 ? 1 : 0;
  }
  const int non_zero_count = negative_count + positive_count;
  if (non_zero_count == 0) {
    return 1.0;
  }
  return std::min(1.0, 2.0 * binomial_half_cumulative(
                               non_zero_count, std::min(negative_count, positive_count)));
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
  std::sort(round_differences.begin(), round_differences.end());
  const OrderStatisticInterval median_interval = interval_for_median(round_differences);
  paired_comparison.difference_interval_low = median_interval.low;
  paired_comparison.difference_interval_high = median_interval.high;
  paired_comparison.difference_interval_coverage = median_interval.coverage;
  paired_comparison.sign_test_p_value = sign_test_p_value(round_differences);
  if (median_interval.coverage >= MINIMUM_INTERVAL_COVERAGE) {
    if (median_interval.high < 0.0) {
      paired_comparison.verdict = ComparisonVerdict::candidate_faster;
    } else if (median_interval.low > 0.0) {
      paired_comparison.verdict = ComparisonVerdict::candidate_slower;
    }
  }
  return paired_comparison;
}

} // namespace mlx_perf
