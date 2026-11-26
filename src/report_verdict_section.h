#pragma once

// Verdict section: one concern only — say at the top of a comparison which
// variant is faster, by how much, and whether that difference exceeds the
// run-to-run spread, for job duration, GPU execution and host submission.

#include "output_agreement.h"
#include "paired_measurement.h"
#include "sample_statistics.h"

#include <string>

namespace mlx_perf {

// The one-sentence answer for one metric, as plain text (escaped when
// rendered). Lower is better, so "faster" means a smaller median.
std::string describe_verdict(const std::string& baseline_name, const std::string& candidate_name,
                             const PairedComparison& paired_comparison);

// The headline sentence (job duration), the per-metric table with each
// median beside the middle half of its rounds (the range the verdict is
// judged on), the rounds the candidate won, and how closely the two
// variants' outputs agree.
std::string render_verdict_section(const std::string& baseline_name,
                                   const std::string& candidate_name,
                                   const PairComparisons& pair_comparisons,
                                   const OutputAgreement& output_agreement);

} // namespace mlx_perf
