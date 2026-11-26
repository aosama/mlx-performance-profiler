// Component tests for comparison_record: the comparison as JSON, from a
// synthetic pair (no GPU), written and read back as a reader would.

#include "comparison_record.h"
#include "json_value.h"
#include "json_writer.h"
#include "report_fixture.h"
#include "test_harness.h"

#include <string>

namespace {

using namespace mlx_perf;

JsonValue written_and_parsed(const PairedMeasurement& paired_measurement) {
  return parse_json(write_json(build_comparison_record(paired_measurement, "base_stock.json",
                                                       "base_custom.json")));
}

// --- positive ---

void should_name_both_variants_and_their_measurement_files() {
  PairedMeasurement paired_measurement;
  report_fixture::populate_pair(paired_measurement);
  const JsonValue record = written_and_parsed(paired_measurement);
  EXPECT_TRUE(record.at("kind").as_string() == "comparison");
  EXPECT_TRUE(record.at("schema_version").as_number() == 1);
  EXPECT_TRUE(record.at("baseline").at("variant_name").as_string() == "stock");
  EXPECT_TRUE(record.at("candidate").at("measurement_file").as_string() == "base_custom.json");
}

void should_carry_each_metrics_samples_interval_and_verdict() {
  PairedMeasurement paired_measurement;
  report_fixture::populate_pair(paired_measurement);
  const JsonValue record = written_and_parsed(paired_measurement);
  const JsonValue& gpu_span = record.at("metrics").at("gpu_span_us");
  EXPECT_TRUE(gpu_span.at("verdict").as_string() == "candidate_faster");
  EXPECT_TRUE(gpu_span.at("median_difference_us").as_number() == -10);
  EXPECT_TRUE(gpu_span.at("rounds_candidate_faster").as_number() == 12);
  EXPECT_TRUE(gpu_span.at("baseline").at("samples").as_array().size() == 12);
  EXPECT_TRUE(gpu_span.at("candidate").at("min").as_number() == 40);
  EXPECT_TRUE(gpu_span.at("difference_interval_us").at("high").as_number() == -10);
  EXPECT_NEAR(gpu_span.at("difference_interval_us").at("coverage").as_number(),
              1.0 - 2.0 * 79.0 / 4096.0, 1e-12);
  EXPECT_NEAR(gpu_span.at("sign_test_p_value").as_number(), 2.0 / 4096.0, 1e-12);
  const JsonValue& job_wall = record.at("metrics").at("job_wall_us");
  EXPECT_TRUE(job_wall.at("verdict").as_string() == "within_spread");
  EXPECT_TRUE(job_wall.at("rounds_candidate_faster").as_number() == 6);
  EXPECT_TRUE(record.at("metrics").at("host_submission_us").at("round_count").as_number() == 12);
  EXPECT_TRUE(record.at("output_agreement").at("max_absolute_difference").as_number() == 0.25);
}

// --- edge / boundary ---

void should_record_a_single_round_comparison() {
  PairedMeasurement paired_measurement;
  report_fixture::populate_pair(paired_measurement);
  paired_measurement.baseline_rounds = {{10}, {5}, {20}, {}};
  paired_measurement.candidate_rounds = {{10}, {5}, {20}, {}};
  const JsonValue record = written_and_parsed(paired_measurement);
  EXPECT_TRUE(record.at("metrics").at("job_wall_us").at("median_difference_us").as_number() ==
              0);
  EXPECT_TRUE(record.at("metrics").at("job_wall_us").at("verdict").as_string() ==
              "within_spread");
}

void should_judge_memory_in_bytes_beside_the_rounds_pressure() {
  PairedMeasurement paired_measurement;
  report_fixture::populate_pair_with_memory(paired_measurement);
  const JsonValue record = written_and_parsed(paired_measurement);
  const JsonValue& peak_working_set = record.at("memory").at("metrics").at("peak_mlx_active_added_bytes");
  EXPECT_TRUE(peak_working_set.at("verdict").as_string() == "candidate_slower");
  EXPECT_TRUE(peak_working_set.at("median_difference_bytes").as_number() == 256.0 * 1048576.0);
  EXPECT_TRUE(peak_working_set.at("difference_interval_bytes").at("low").as_number() > 0);
  EXPECT_TRUE(record.at("memory").at("candidate").at("rounds_crossing_wire_limit").as_number() == 1);
  EXPECT_TRUE(record.at("memory").at("baseline").at("rounds").as_array().size() == 12);
}

void should_write_null_memory_when_the_sides_rounds_do_not_pair_up() {
  PairedMeasurement paired_measurement;
  report_fixture::populate_pair_with_memory(paired_measurement);
  paired_measurement.candidate_rounds.memory_figures.pop_back();
  EXPECT_TRUE(written_and_parsed(paired_measurement).at("memory").is_null());
}

void should_write_null_memory_for_rounds_without_it() {
  PairedMeasurement paired_measurement;
  report_fixture::populate_pair(paired_measurement);
  EXPECT_TRUE(written_and_parsed(paired_measurement).at("memory").is_null());
}

// --- negative ---

void should_refuse_a_pair_without_rounds() {
  PairedMeasurement paired_measurement;
  report_fixture::populate_pair(paired_measurement);
  paired_measurement.baseline_rounds = {};
  paired_measurement.candidate_rounds = {};
  bool has_refused = false;
  try {
    build_comparison_record(paired_measurement, "a.json", "b.json");
  } catch (const std::invalid_argument&) {
    has_refused = true;
  }
  EXPECT_TRUE(has_refused);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_name_both_variants_and_their_measurement_files),
      TEST_CASE(should_carry_each_metrics_samples_interval_and_verdict),
      TEST_CASE(should_record_a_single_round_comparison),
      TEST_CASE(should_judge_memory_in_bytes_beside_the_rounds_pressure),
      TEST_CASE(should_write_null_memory_for_rounds_without_it),
      TEST_CASE(should_write_null_memory_when_the_sides_rounds_do_not_pair_up),
      TEST_CASE(should_refuse_a_pair_without_rounds),
  });
}
