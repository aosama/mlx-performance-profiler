#pragma once

// Measurement file: one concern only — persist one measured variant as
// `<output_directory>/<file_base>.json`. The document's content is
// measurement_document's; this module only names and writes the file.

#include "measurement_session.h"

#include <string>

namespace mlx_perf {

// Returns the path written, or an empty string when the file could not be
// written (missing directory, permissions, full disk).
std::string write_measurement_file(const VariantMeasurement& variant_measurement,
                                   const std::string& output_directory,
                                   const std::string& file_base);

} // namespace mlx_perf
