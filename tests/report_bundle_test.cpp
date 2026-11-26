// Component tests for report_bundle: a paired measurement written as three
// reports, two measurement files and the comparison record, from synthetic
// measurements (no GPU). Runs in the build directory, where CTest starts it.

#include "json_value.h"
#include "paired_measurement.h"
#include "report_bundle.h"
#include "report_fixture.h"
#include "test_harness.h"

#include <fstream>
#include <iterator>
#include <string>

namespace {

using mlx_perf::PairedMeasurement;
using mlx_perf::ReportBundleRequest;

bool file_exists(const std::string& file_path) { return std::ifstream(file_path).good(); }

std::string read_file(const std::string& file_path) {
  std::ifstream input_stream(file_path);
  return std::string(std::istreambuf_iterator<char>(input_stream), {});
}

// --- positive ---

void should_write_three_reports_two_measurement_files_and_the_comparison_record() {
  PairedMeasurement paired_measurement;
  report_fixture::populate_pair(paired_measurement);
  const ReportBundleRequest bundle_request{"Bundle test", ".", "bundle_test"};
  EXPECT_TRUE(mlx_perf::write_report_bundle(bundle_request, paired_measurement));
  for (const char* written_file :
       {"bundle_test.html", "bundle_test_stock.html", "bundle_test_custom.html",
        "bundle_test_stock.json", "bundle_test_custom.json", "bundle_test.json"}) {
    EXPECT_TRUE(file_exists(written_file));
  }
  const mlx_perf::JsonValue candidate_document =
      mlx_perf::parse_json_file("bundle_test_custom.json");
  EXPECT_TRUE(candidate_document.at("variant_name").as_string() == "custom");
  const mlx_perf::JsonValue comparison_record = mlx_perf::parse_json_file("bundle_test.json");
  EXPECT_TRUE(comparison_record.at("baseline").at("measurement_file").as_string() ==
              "bundle_test_stock.json");
  // The comparison page opens with the verdict.
  EXPECT_TRUE(read_file("bundle_test.html").find("id=\"verdict\"") != std::string::npos);
}

// --- edge / boundary ---

void should_write_a_pair_of_empty_variants_without_failing() {
  PairedMeasurement paired_measurement;
  paired_measurement.baseline.variant_name = "first";
  paired_measurement.candidate.variant_name = "second";
  paired_measurement.baseline_rounds = {{1}, {0}, {1}};
  paired_measurement.candidate_rounds = {{1}, {0}, {1}};
  const ReportBundleRequest bundle_request{"Empty", ".", "bundle_test_empty"};
  EXPECT_TRUE(mlx_perf::write_report_bundle(bundle_request, paired_measurement));
  EXPECT_TRUE(
      mlx_perf::parse_json_file("bundle_test_empty_second.json").at("nodes").as_array().empty());
}

// --- negative ---

void should_refuse_variant_names_that_differ_only_in_case() {
  PairedMeasurement paired_measurement;
  report_fixture::populate_pair(paired_measurement);
  paired_measurement.baseline.variant_name = "Stock";
  paired_measurement.candidate.variant_name = "stock";
  const ReportBundleRequest bundle_request{"Collide", ".", "bundle_test_collide"};
  EXPECT_TRUE(!mlx_perf::write_report_bundle(bundle_request, paired_measurement));
  EXPECT_TRUE(!file_exists("bundle_test_collide.html"));
}

void should_report_failure_when_the_directory_is_missing() {
  PairedMeasurement paired_measurement;
  report_fixture::populate_pair(paired_measurement);
  const ReportBundleRequest bundle_request{"Missing", "no/such/directory", "x"};
  EXPECT_TRUE(!mlx_perf::write_report_bundle(bundle_request, paired_measurement));
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_write_three_reports_two_measurement_files_and_the_comparison_record),
      TEST_CASE(should_write_a_pair_of_empty_variants_without_failing),
      TEST_CASE(should_refuse_variant_names_that_differ_only_in_case),
      TEST_CASE(should_report_failure_when_the_directory_is_missing),
  });
}
