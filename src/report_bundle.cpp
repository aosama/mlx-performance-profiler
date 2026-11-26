#include "report_bundle.h"

#include "measurement_file.h"
#include "report_comparison_page.h"
#include "report_writer.h"

#include <cstdio>

namespace mlx_perf {
namespace {

std::string variant_file_base(const ReportBundleRequest& bundle_request,
                              const VariantMeasurement& variant_measurement) {
  return bundle_request.file_base + "_" + variant_measurement.variant_name;
}

ReportPaths write_variant(const ReportBundleRequest& bundle_request,
                          const VariantMeasurement& variant_measurement) {
  const ReportInputs report_inputs{
      bundle_request.job_title + " (" + variant_measurement.variant_name + ")",
      variant_measurement.timeline_graph, variant_measurement.gpu_timeline,
      variant_measurement.serial_telemetry, variant_measurement.job_totals};
  const ReportPaths report_paths =
      write_report(report_inputs, bundle_request.output_directory,
                   variant_file_base(bundle_request, variant_measurement));
  if (!report_paths.is_written) {
    std::fprintf(stderr, "error: could not write %s\n", report_paths.html_path.c_str());
  } else {
    std::printf("report: %s%s\n", report_paths.html_path.c_str(),
                report_paths.graphviz_available ? " (with SVG)" : " (no Graphviz)");
  }
  return report_paths;
}

// The same measurement as data, beside its report, for callers that assert
// on or plot numbers rather than read a page.
bool write_variant_measurements(const ReportBundleRequest& bundle_request,
                                const VariantMeasurement& variant_measurement) {
  const std::string measurement_path =
      write_measurement_file(variant_measurement, bundle_request.output_directory,
                             variant_file_base(bundle_request, variant_measurement));
  if (measurement_path.empty()) {
    std::fprintf(stderr, "error: could not write the measurements of %s\n",
                 variant_measurement.variant_name.c_str());
    return false;
  }
  std::printf("measurements: %s\n", measurement_path.c_str());
  return true;
}

} // namespace

bool write_report_bundle(const ReportBundleRequest& bundle_request,
                         const VariantMeasurement& baseline,
                         const VariantMeasurement& candidate) {
  const ReportPaths baseline_paths = write_variant(bundle_request, baseline);
  const ReportPaths candidate_paths = write_variant(bundle_request, candidate);
  // Both written before either result is read: one failed file must not
  // stop the other from being attempted.
  const bool is_baseline_measurement_written =
      write_variant_measurements(bundle_request, baseline);
  const bool is_candidate_measurement_written =
      write_variant_measurements(bundle_request, candidate);
  const bool are_measurements_written =
      is_baseline_measurement_written && is_candidate_measurement_written;
  const ComparedVariant baseline_variant{
      baseline, baseline_paths.plan_svg, variant_file_base(bundle_request, baseline) + ".html"};
  const ComparedVariant candidate_variant{
      candidate, candidate_paths.plan_svg,
      variant_file_base(bundle_request, candidate) + ".html"};
  const std::string comparison_path =
      bundle_request.output_directory + "/" + bundle_request.file_base + ".html";
  const bool is_comparison_written = write_text_file(
      comparison_path,
      build_comparison_document(bundle_request.job_title + ": " + baseline.variant_name +
                                    " vs " + candidate.variant_name,
                                baseline_variant, candidate_variant));
  if (!is_comparison_written) {
    std::fprintf(stderr, "error: could not write %s\n", comparison_path.c_str());
  } else {
    std::printf("report: %s (%s vs %s)\n", comparison_path.c_str(),
                baseline.variant_name.c_str(), candidate.variant_name.c_str());
  }
  return baseline_paths.is_written && candidate_paths.is_written && is_comparison_written &&
         are_measurements_written;
}

} // namespace mlx_perf
