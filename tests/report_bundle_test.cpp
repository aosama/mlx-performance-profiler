// Component tests for report_bundle: a baseline/candidate pair written as
// three reports and two measurement files, from synthetic measurements (no
// GPU). Runs in the build directory, where CTest starts it.

#include "json_value.h"
#include "measurement_session.h"
#include "report_bundle.h"
#include "report_fixture.h"
#include "test_harness.h"

#include <fstream>
#include <string>

namespace {

using mlx_perf::ReportBundleRequest;
using mlx_perf::VariantMeasurement;

bool file_exists(const std::string& file_path) { return std::ifstream(file_path).good(); }

void populate_variant(VariantMeasurement& variant_measurement, const char* variant_name) {
  report_fixture::populate_two_stream_run(variant_measurement);
  variant_measurement.variant_name = variant_name;
}

// --- positive ---

void should_write_three_reports_and_a_measurement_file_per_variant() {
  VariantMeasurement baseline;
  VariantMeasurement candidate;
  populate_variant(baseline, "eager");
  populate_variant(candidate, "compiled");
  const ReportBundleRequest bundle_request{"Bundle test", ".", "bundle_test"};
  EXPECT_TRUE(mlx_perf::write_report_bundle(bundle_request, baseline, candidate));
  for (const char* written_file :
       {"bundle_test.html", "bundle_test_eager.html", "bundle_test_compiled.html",
        "bundle_test_eager.json", "bundle_test_compiled.json"}) {
    EXPECT_TRUE(file_exists(written_file));
  }
  const mlx_perf::JsonValue candidate_document =
      mlx_perf::parse_json_file("bundle_test_compiled.json");
  EXPECT_TRUE(candidate_document.at("variant_name").as_string() == "compiled");
}

// --- edge / boundary ---

void should_write_an_empty_pair_without_failing() {
  const VariantMeasurement empty_baseline;
  const VariantMeasurement empty_candidate;
  const ReportBundleRequest bundle_request{"Empty", ".", "bundle_test_empty"};
  EXPECT_TRUE(mlx_perf::write_report_bundle(bundle_request, empty_baseline, empty_candidate));
  // Both variants are unnamed, so both measurement files share one name.
  EXPECT_TRUE(mlx_perf::parse_json_file("bundle_test_empty_.json").at("nodes").as_array().empty());
}

// --- negative ---

void should_report_failure_when_the_directory_is_missing() {
  VariantMeasurement baseline;
  VariantMeasurement candidate;
  populate_variant(baseline, "eager");
  populate_variant(candidate, "compiled");
  const ReportBundleRequest bundle_request{"Missing", "no/such/directory", "x"};
  EXPECT_TRUE(!mlx_perf::write_report_bundle(bundle_request, baseline, candidate));
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_write_three_reports_and_a_measurement_file_per_variant),
      TEST_CASE(should_write_an_empty_pair_without_failing),
      TEST_CASE(should_report_failure_when_the_directory_is_missing),
  });
}
