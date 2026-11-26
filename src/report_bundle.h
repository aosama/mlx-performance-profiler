#pragma once

// Report bundle: one concern only — write a paired measurement as three
// reports (one per variant, plus the comparison page with its verdict), each
// variant's measurement file and the comparison record, reporting each file
// written or failed on stdout/stderr.

#include "paired_measurement.h"

#include <string>

namespace mlx_perf {

struct ReportBundleRequest {
  std::string job_title;
  std::string output_directory;
  // Writes <base>.html and <base>.json (the comparison), and
  // <base>_<variant>.html and .json per variant.
  std::string file_base;
};

// Returns false if any report or measurement file could not be written.
bool write_report_bundle(const ReportBundleRequest& bundle_request,
                         const PairedMeasurement& paired_measurement);

} // namespace mlx_perf
