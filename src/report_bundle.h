#pragma once

// Report bundle: one concern only — write a baseline/candidate pair of
// measured variants as three reports (one per variant plus the comparison
// page), reporting each file written or failed on stdout/stderr.

#include "measurement_session.h"

#include <string>

namespace mlx_perf {

struct ReportBundleRequest {
  std::string job_title;
  std::string output_directory;
  std::string file_base;  // writes <base>.html, <base>_<variant>.html
};

// Returns false if any of the three reports could not be written.
bool write_report_bundle(const ReportBundleRequest& bundle_request,
                         const VariantMeasurement& baseline,
                         const VariantMeasurement& candidate);

} // namespace mlx_perf
