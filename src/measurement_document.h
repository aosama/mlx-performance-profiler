#pragma once

// Measurement document: one concern only — express one measured variant as
// a JSON tree (schema version 1, described in docs/measurement-file.md). Pure:
// no files and no GPU, so every field is tested from synthetic measurements.
// The HTML report and this document read the same VariantMeasurement in the
// same call, so the two can never disagree.

#include "json_value.h"
#include "measurement_session.h"

namespace mlx_perf {

// Bumped on any change a reader of an older document would misread:
// a removed or renamed field, or a changed meaning or unit.
inline constexpr int MEASUREMENT_SCHEMA_VERSION = 2;

// Values that were not measured (a CPU node's GPU window, a stage member's
// own timing, a serial duration whose node did not match) are null, never 0.
JsonValue build_measurement_document(const VariantMeasurement& variant_measurement);

} // namespace mlx_perf
