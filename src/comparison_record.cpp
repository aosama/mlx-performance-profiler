#include "comparison_record.h"

#include "measurement_document.h"

#include <string>
#include <vector>

namespace mlx_perf {
namespace {

using Object = JsonValue::Object;
using Array = JsonValue::Array;


JsonValue sample_record(const SampleSummary& sample_summary,
                        const std::vector<double>& samples) {
  Array sample_values;
  for (const double sample_value : samples) {
    sample_values.push_back(json_number(sample_value));
  }
  return JsonValue(Object{{"median", json_number(sample_summary.median)},
                          {"min", json_number(sample_summary.minimum)},
                          {"max", json_number(sample_summary.maximum)},
                          {"lower_quartile", json_number(sample_summary.lower_quartile)},
                          {"upper_quartile", json_number(sample_summary.upper_quartile)},
                          {"samples", JsonValue(std::move(sample_values))}});
}

JsonValue variant_reference(const VariantMeasurement& variant_measurement,
                            const std::string& measurement_file) {
  return JsonValue(Object{{"variant_name", json_text(variant_measurement.variant_name)},
                          {"measurement_file", json_text(measurement_file)}});
}

} // namespace

JsonValue build_metric_record(const PairedComparison& paired_comparison,
                        const std::vector<double>& baseline_samples,
                        const std::vector<double>& candidate_samples) {
  return JsonValue(Object{
      {"baseline", sample_record(paired_comparison.baseline, baseline_samples)},
      {"candidate", sample_record(paired_comparison.candidate, candidate_samples)},
      {"round_difference_us", JsonValue(Object{
          {"median", json_number(paired_comparison.round_difference.median)},
          {"min", json_number(paired_comparison.round_difference.minimum)},
          {"max", json_number(paired_comparison.round_difference.maximum)}})},
      {"difference_interval_us", JsonValue(Object{
          {"low", json_number(paired_comparison.difference_interval_low)},
          {"high", json_number(paired_comparison.difference_interval_high)},
          {"coverage", json_number(paired_comparison.difference_interval_coverage)}})},
      {"sign_test_p_value", json_number(paired_comparison.sign_test_p_value)},
      {"median_difference_us", json_number(paired_comparison.median_difference)},
      {"relative_difference", json_number(paired_comparison.relative_difference)},
      {"verdict", json_text(std::string(comparison_verdict_name(paired_comparison.verdict)))},
      {"rounds_candidate_faster", json_number(paired_comparison.rounds_candidate_faster)},
      {"round_count", json_number(paired_comparison.round_count)}});
}

JsonValue build_comparison_record(const PairedMeasurement& paired_measurement,
                                  const std::string& baseline_measurement_file,
                                  const std::string& candidate_measurement_file) {
  const PairComparisons pair_comparisons = compare_pair(paired_measurement);
  const VariantRounds& baseline_rounds = paired_measurement.baseline_rounds;
  const VariantRounds& candidate_rounds = paired_measurement.candidate_rounds;
  return JsonValue(Object{
      {"schema_version", json_number(MEASUREMENT_SCHEMA_VERSION)},
      {"kind", json_text("comparison")},
      {"baseline", variant_reference(paired_measurement.baseline, baseline_measurement_file)},
      {"candidate",
       variant_reference(paired_measurement.candidate, candidate_measurement_file)},
      {"metrics", JsonValue(Object{
          {"job_wall_us", build_metric_record(pair_comparisons.job_wall,
                                        baseline_rounds.job_wall_microseconds,
                                        candidate_rounds.job_wall_microseconds)},
          {"gpu_span_us", build_metric_record(pair_comparisons.gpu_span,
                                        baseline_rounds.gpu_span_microseconds,
                                        candidate_rounds.gpu_span_microseconds)},
          {"host_submission_us",
           build_metric_record(pair_comparisons.host_submission,
                         baseline_rounds.host_submission_microseconds,
                         candidate_rounds.host_submission_microseconds)}})},
      {"output_agreement", JsonValue(Object{
          {"output_count", json_number(paired_measurement.output_agreement.output_count)},
          {"max_absolute_difference",
           json_number(paired_measurement.output_agreement.max_absolute_difference)}})}});
}

} // namespace mlx_perf
