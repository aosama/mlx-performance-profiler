#pragma once

// Verdict section: one concern only — say at the top of a comparison which
// variant is faster, by how much, and whether that difference exceeds the
// run-to-run spread, for job duration, GPU execution and host submission.

#include "output_agreement.h"
#include "paired_measurement.h"
#include "sample_statistics.h"

#include <string>
#include <vector>

namespace mlx_perf {

// What a judged metric measures, which decides how it is written: time in
// µs ("faster", "slower") or memory in bytes ("less", "more"). Lower is
// better for both.
enum class MetricUnit { microseconds, bytes };

// The one-sentence answer for one metric, as plain text (escaped when
// rendered). Lower is better, so "faster" means a smaller median.
std::string describe_verdict(const std::string& baseline_name, const std::string& candidate_name,
                             const PairedComparison& paired_comparison,
                             MetricUnit metric_unit = MetricUnit::microseconds);

// One judged metric for render_verdict_table.
struct NamedComparison {
  const char* metric_name;
  const PairedComparison* paired_comparison;
};

// The table under a verdict: per metric, each variant's median beside the
// middle half of its own rounds, the median per-round difference, the
// interval the verdict was judged on, the verdict and the rounds won.
// Every row of one table shares `table_unit`, so its rounds column reads
// "was faster" (time) or "used less" (memory).
std::string render_verdict_table(const std::string& baseline_name, const std::string& candidate_name,
                                 const std::vector<NamedComparison>& named_comparisons,
                                 MetricUnit table_unit = MetricUnit::microseconds);

// The headline sentence (job duration), then per metric: each variant's
// median beside the middle half of its own rounds, the median per-round
// difference beside the middle half of those differences (what the verdict
// is judged on), the verdict, the rounds the candidate won, and how closely
// the two variants' outputs agree.
//
// With each side's rounds, the section also says under the headline when
// rounds crossed the wire limit or saw the system compress memory; when the
// memory comparisons were measured, it ends with the judged memory table.
std::string render_verdict_section(const std::string& baseline_name,
                                   const std::string& candidate_name,
                                   const PairComparisons& pair_comparisons,
                                   const OutputAgreement& output_agreement,
                                   const VariantRounds* baseline_rounds = nullptr,
                                   const VariantRounds* candidate_rounds = nullptr);

} // namespace mlx_perf
