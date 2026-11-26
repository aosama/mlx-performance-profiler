#pragma once

// Comparison page: one concern only — the "planned vs executed" view of two
// implementations of the same job (eager vs compiled, or a stock primitive
// vs a custom kernel): the verdict when the caller supplies one, a metric
// delta table, the two execution graphs side by side, both timelines on one shared time
// scale, and the primitive mix (with what compile fused, when it did).

#include "measurement_session.h"

#include <string>

namespace mlx_perf {

struct ComparedVariant {
  const VariantMeasurement& variant_measurement;
  // The variant's measurement document as text (measurement_document.h),
  // drawn by the interactive graph view.
  std::string measurement_document_json;
  std::string full_report_file;  // relative link to the variant's report
};

// The metric delta table alone (also used by the live server's Job tab):
// single figures per variant, labelled as not judged against a spread.
std::string render_comparison_metrics(
    const VariantMeasurement& baseline, const VariantMeasurement& candidate);

// Caller-supplied text above the metrics. An empty subtitle becomes "Same
// job, two implementations: <baseline> vs <candidate>"; an empty verdict is
// omitted (single-pass comparisons have no spread to judge against).
struct ComparisonPageText {
  std::string subtitle;
  std::string verdict_html;
};

std::string build_comparison_document(
    const std::string& page_title,
    const ComparedVariant& baseline_variant,
    const ComparedVariant& candidate_variant,
    const ComparisonPageText& page_text = {});

// "CompiledSigmoidMultiply" → "Sigmoid · Multiply". MLX names a fused node
// "Compiled" + each fused primitive's name; splitting at capitals recovers
// them, approximately for multi-capital names (e.g. "AddMM" never fuses, so
// the approximation does not bite on fusable elementwise ops).
std::string describe_fused_primitives(const std::string& compiled_op_name);

} // namespace mlx_perf
