// Component tests for procedure_comparison_record: a procedure comparison as
// JSON, from a synthetic pair (no GPU), written and read back as a reader
// would.

#include "json_value.h"
#include "json_writer.h"
#include "procedure_comparison_record.h"
#include "procedure_fixture.h"
#include "wired_memory_fixture.h"
#include "measurement_document.h"
#include "test_harness.h"

#include <string>

namespace {

using namespace mlx_perf;

JsonValue written_and_parsed(const PairedProcedureMeasurement& paired_measurement) {
  return parse_json(write_json(
      build_procedure_comparison_record(paired_measurement, compare_procedure_pair(paired_measurement))));
}

const JsonValue* find_function(const JsonValue& procedure_record, const std::string& kernel_function_name) {
  for (const JsonValue& launch_record : procedure_record.at("kernel_launches").as_array()) {
    if (launch_record.at("kernel_function_name").as_string() == kernel_function_name) return &launch_record;
  }
  return nullptr;
}

// --- positive ---

void should_name_the_kind_the_gpu_and_both_procedures() {
  const JsonValue record = written_and_parsed(procedure_fixture::make_chatty_vs_batched());
  EXPECT_TRUE(record.at("kind").as_string() == "procedure_comparison");
  EXPECT_TRUE(record.at("schema_version").as_number() == mlx_perf::MEASUREMENT_SCHEMA_VERSION);
  EXPECT_TRUE(record.at("gpu").at("name").as_string() == "Apple M5 Pro");
  EXPECT_TRUE(record.at("gpu").at("core_count").as_number() == 16);
  EXPECT_TRUE(record.at("baseline").at("procedure_name").as_string() == "chatty");
  EXPECT_TRUE(record.at("candidate").at("procedure_name").as_string() == "batched");
}

void should_carry_every_round_and_the_observed_run() {
  const JsonValue record = written_and_parsed(procedure_fixture::make_chatty_vs_batched());
  const JsonValue::Array& baseline_rounds = record.at("baseline").at("rounds").as_array();
  EXPECT_TRUE(baseline_rounds.size() == 12);
  EXPECT_TRUE(baseline_rounds[3].at("duration_us").as_number() == 403);
  EXPECT_TRUE(baseline_rounds[3].at("gpu_idle_us").as_number() == 303);
  EXPECT_TRUE(baseline_rounds[3].at("blocking_wait_count").as_number() == 8);
  EXPECT_TRUE(baseline_rounds[3].at("blocking_wait_us").as_number() == 80);
  EXPECT_TRUE(record.at("candidate").at("observed_run").at("command_buffer_count").as_number() == 1);
}

void should_judge_duration_busy_and_idle_separately() {
  const JsonValue record = written_and_parsed(procedure_fixture::make_chatty_vs_batched());
  const JsonValue& metrics = record.at("metrics");
  EXPECT_TRUE(metrics.at("duration_us").at("verdict").as_string() == "candidate_faster");
  EXPECT_TRUE(metrics.at("duration_us").at("median_difference_us").as_number() == -250);
  EXPECT_TRUE(metrics.at("gpu_idle_us").at("verdict").as_string() == "candidate_faster");
  EXPECT_TRUE(metrics.at("gpu_busy_us").at("verdict").as_string() == "within_spread");
  EXPECT_TRUE(metrics.at("duration_us").at("baseline").at("samples").as_array().size() == 12);
}

void should_judge_memory_in_bytes_and_carry_each_rounds_figures() {
  const JsonValue record = written_and_parsed(procedure_fixture::make_chatty_vs_batched());
  const JsonValue& peak_working_set = record.at("metrics").at("peak_mlx_active_added_bytes");
  EXPECT_TRUE(peak_working_set.at("verdict").as_string() == "candidate_slower");
  EXPECT_TRUE(peak_working_set.at("median_difference_bytes").as_number() == 256.0 * 1048576.0);
  EXPECT_TRUE(record.at("metrics").at("mlx_held_change_bytes").at("verdict").as_string() == "within_spread");
  EXPECT_TRUE(record.at("baseline").at("rounds").as_array().at(0).at("memory").at("peak_mlx_active_added_bytes")
                  .as_number() == 64.0 * 1048576.0);
  EXPECT_TRUE(record.at("candidate").at("rounds_crossing_wire_limit").as_number() == 0);
  EXPECT_TRUE(record.at("candidate").at("rounds_with_compression").as_number() == 0);
}

void should_group_launches_by_metal_function_and_flag_the_underfilling_ones() {
  const JsonValue record = written_and_parsed(procedure_fixture::make_chatty_vs_batched());
  EXPECT_TRUE(record.at("baseline").at("kernel_launch_count").as_number() == 8);
  const JsonValue* add_launches = find_function(record.at("baseline"), "v_Add_float32");
  EXPECT_TRUE(add_launches != nullptr && add_launches->at("launch_count").as_number() == 8);
  EXPECT_TRUE(add_launches != nullptr && add_launches->at("has_launch_unable_to_fill_gpu").as_bool());
  EXPECT_TRUE(add_launches != nullptr && add_launches->at("lowest_gpu_core_coverage").as_number() == 0.125);
  const JsonValue* gemv_launches = find_function(record.at("candidate"), "gemv_float32");
  EXPECT_TRUE(gemv_launches != nullptr && !gemv_launches->at("has_launch_unable_to_fill_gpu").as_bool());
  EXPECT_TRUE(gemv_launches != nullptr && gemv_launches->at("most_threadgroups").as_number() == 64);
}

// --- edge / boundary ---

void should_write_null_coverage_and_gpu_facts_when_the_gpu_is_unknown() {
  PairedProcedureMeasurement paired_measurement = procedure_fixture::make_chatty_vs_batched();
  paired_measurement.gpu_device_facts = {};
  const JsonValue record = written_and_parsed(paired_measurement);
  EXPECT_TRUE(record.at("gpu").at("name").is_null());
  EXPECT_TRUE(record.at("gpu").at("core_count").is_null());
  const JsonValue* add_launches = find_function(record.at("baseline"), "v_Add_float32");
  EXPECT_TRUE(add_launches != nullptr && add_launches->at("lowest_gpu_core_coverage").is_null());
  EXPECT_TRUE(add_launches != nullptr && add_launches->at("has_launch_unable_to_fill_gpu").is_null());
}

void should_name_launches_without_a_metal_function_as_unnamed() {
  PairedProcedureMeasurement paired_measurement = procedure_fixture::make_chatty_vs_batched();
  paired_measurement.candidate.observed_launches = {procedure_fixture::launch_of("", 16)};
  const JsonValue record = written_and_parsed(paired_measurement);
  EXPECT_TRUE(find_function(record.at("candidate"), "(unnamed pipeline)") != nullptr);
}

void should_record_a_procedure_that_launched_nothing() {
  PairedProcedureMeasurement paired_measurement = procedure_fixture::make_chatty_vs_batched();
  paired_measurement.candidate.observed_launches.clear();
  const JsonValue record = written_and_parsed(paired_measurement);
  EXPECT_TRUE(record.at("candidate").at("kernel_launch_count").as_number() == 0);
  EXPECT_TRUE(record.at("candidate").at("kernel_launches").as_array().empty());
}

void should_write_each_procedures_wired_memory_or_null_without_samples() {
  PairedProcedureMeasurement paired_measurement = procedure_fixture::make_chatty_vs_batched();
  paired_measurement.baseline.observed_wired_memory = wired_memory_fixture::release_then_unwire();
  const JsonValue record = written_and_parsed(paired_measurement);
  EXPECT_TRUE(record.at("baseline").at("wired_memory").at("sample_count").as_number() == 8);
  EXPECT_TRUE(record.at("baseline").at("wired_memory").at("release_events").as_array().size() == 1);
  EXPECT_TRUE(record.at("candidate").at("wired_memory").is_null());
}

// --- negative ---

void should_judge_too_few_rounds_within_spread() {
  const JsonValue record = written_and_parsed(procedure_fixture::make_chatty_vs_batched(3));
  EXPECT_TRUE(record.at("metrics").at("duration_us").at("verdict").as_string() == "within_spread");
  EXPECT_TRUE(record.at("metrics").at("duration_us").at("round_count").as_number() == 3);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_name_the_kind_the_gpu_and_both_procedures),
      TEST_CASE(should_carry_every_round_and_the_observed_run),
      TEST_CASE(should_judge_duration_busy_and_idle_separately),
      TEST_CASE(should_judge_memory_in_bytes_and_carry_each_rounds_figures),
      TEST_CASE(should_group_launches_by_metal_function_and_flag_the_underfilling_ones),
      TEST_CASE(should_write_null_coverage_and_gpu_facts_when_the_gpu_is_unknown),
      TEST_CASE(should_name_launches_without_a_metal_function_as_unnamed),
      TEST_CASE(should_record_a_procedure_that_launched_nothing),
      TEST_CASE(should_write_each_procedures_wired_memory_or_null_without_samples),
      TEST_CASE(should_judge_too_few_rounds_within_spread),
  });
}
