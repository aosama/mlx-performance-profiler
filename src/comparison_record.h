#pragma once

// Comparison record: one concern only — express a paired comparison as a
// JSON tree (`"kind": "comparison"`, schema version 1, described in
// docs/measurement-file.md). Each variant's per-node detail lives in its own
// measurement file, which this record names rather than repeats. Pure: no
// files and no GPU.

#include "json_value.h"
#include "paired_measurement.h"

#include <string>
#include <vector>

namespace mlx_perf {

// One judged metric as JSON: each side's summary and samples, the per-round
// differences, the interval, the sign test and the verdict. Shared by the
// graph comparison record and the procedure comparison record.
JsonValue build_metric_record(const PairedComparison& paired_comparison,
                              const std::vector<double>& baseline_samples,
                              const std::vector<double>& candidate_samples);

JsonValue build_comparison_record(const PairedMeasurement& paired_measurement,
                                  const std::string& baseline_measurement_file,
                                  const std::string& candidate_measurement_file);

} // namespace mlx_perf
