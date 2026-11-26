#pragma once

// Measurement file: one concern only — persist measurement documents as
// `<output_directory>/<file_base>.json`. What a document says is decided by
// measurement_document and comparison_record; this module only names and
// writes files.

#include "json_value.h"
#include "measurement_session.h"

#include <string>

namespace mlx_perf {

// Each returns the path written, or an empty string when the file could not
// be written (missing directory, permissions, full disk).
std::string write_json_file(const JsonValue& document, const std::string& output_directory,
                            const std::string& file_base);

std::string write_measurement_file(const VariantMeasurement& variant_measurement,
                                   const std::string& output_directory,
                                   const std::string& file_base);

} // namespace mlx_perf
