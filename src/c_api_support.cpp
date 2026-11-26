#include "c_api_support.h"

#include <exception>
#include <stdexcept>
#include <string>

namespace mlx_perf {
namespace {

thread_local std::string last_error_message;

int to_c_verdict(ComparisonVerdict verdict) {
  switch (verdict) {
    case ComparisonVerdict::candidate_faster: return MLX_PERF_VERDICT_CANDIDATE_FASTER;
    case ComparisonVerdict::candidate_slower: return MLX_PERF_VERDICT_CANDIDATE_SLOWER;
    case ComparisonVerdict::within_spread: break;
  }
  return MLX_PERF_VERDICT_WITHIN_SPREAD;
}

} // namespace

void clear_last_c_api_error() { last_error_message.clear(); }

const char* last_c_api_error() { return last_error_message.c_str(); }

// Exceptions must not cross into C or Rust frames (undefined behavior), so
// every entry point catches everything and reports through here.
int record_failure_in_flight() {
  try {
    throw;
  } catch (const std::exception& failure) {
    last_error_message = failure.what();
  } catch (...) {
    last_error_message = "unknown failure";
  }
  return 1;
}

void require_strings(std::initializer_list<const char*> required_strings, const char* entry_point_name) {
  for (const char* required_string : required_strings) {
    if (required_string == nullptr) {
      throw std::invalid_argument(std::string(entry_point_name) + ": null argument");
    }
  }
}

void fill_metric_comparison(const PairedComparison& paired_comparison,
                            mlx_perf_metric_comparison& metric_comparison) {
  metric_comparison.baseline_median_us = paired_comparison.baseline.median;
  metric_comparison.baseline_min_us = paired_comparison.baseline.minimum;
  metric_comparison.baseline_max_us = paired_comparison.baseline.maximum;
  metric_comparison.baseline_lower_quartile_us = paired_comparison.baseline.lower_quartile;
  metric_comparison.baseline_upper_quartile_us = paired_comparison.baseline.upper_quartile;
  metric_comparison.candidate_median_us = paired_comparison.candidate.median;
  metric_comparison.candidate_min_us = paired_comparison.candidate.minimum;
  metric_comparison.candidate_max_us = paired_comparison.candidate.maximum;
  metric_comparison.candidate_lower_quartile_us = paired_comparison.candidate.lower_quartile;
  metric_comparison.candidate_upper_quartile_us = paired_comparison.candidate.upper_quartile;
  metric_comparison.median_difference_us = paired_comparison.median_difference;
  metric_comparison.difference_interval_low_us = paired_comparison.difference_interval_low;
  metric_comparison.difference_interval_high_us = paired_comparison.difference_interval_high;
  metric_comparison.difference_interval_coverage = paired_comparison.difference_interval_coverage;
  metric_comparison.sign_test_p_value = paired_comparison.sign_test_p_value;
  metric_comparison.relative_difference = paired_comparison.relative_difference;
  metric_comparison.verdict = to_c_verdict(paired_comparison.verdict);
  metric_comparison.rounds_candidate_faster = paired_comparison.rounds_candidate_faster;
  metric_comparison.round_count = paired_comparison.round_count;
}

} // namespace mlx_perf
