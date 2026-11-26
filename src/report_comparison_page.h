#pragma once

// Comparison page: one concern only — the "planned vs executed" view of two
// variants of the same job (eager vs compiled): a metric delta table, the two
// DAGs side by side, both timelines on one shared time scale, and the
// primitive mix showing what compile fused.

#include "measurement_session.h"

#include <string>

namespace mlx_perf {

struct ComparedVariant {
  const VariantMeasurement& variant_measurement;
  std::string plan_svg;          // rendered DAG, empty without Graphviz
  std::string full_report_file;  // relative link to the variant's report
};

// The metric delta table alone (also used by the live server's Job tab).
std::string render_comparison_metrics(
    const VariantMeasurement& baseline, const VariantMeasurement& candidate);

std::string build_comparison_document(
    const std::string& page_title,
    const ComparedVariant& baseline_variant,
    const ComparedVariant& candidate_variant);

// "CompiledSigmoidMultiply" → "Sigmoid · Multiply". MLX names a fused node
// "Compiled" + each fused primitive's name; splitting at capitals recovers
// them, approximately for multi-capital names (e.g. "AddMM" never fuses, so
// the approximation does not bite on fusable elementwise ops).
std::string describe_fused_primitives(const std::string& compiled_op_name);

} // namespace mlx_perf
