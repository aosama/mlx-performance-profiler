// Component tests for measurement_document: the JSON tree built from a
// synthetic two-stream measurement (report_fixture.h), no GPU needed. Each
// document is also written and parsed back, as a reader in another language
// would see it.

#include "json_value.h"
#include "json_writer.h"
#include "measurement_document.h"
#include "report_fixture.h"
#include "test_harness.h"

#include <cmath>
#include <string>
#include <utility>

namespace {

using mlx_perf::JsonValue;
using mlx_perf::VariantMeasurement;

constexpr double SLICE_SUM_TOLERANCE_MICROSECONDS = 1e-9;

// Returned by value: callers keep the document alive while they hold
// references into it.
JsonValue written_and_parsed(const VariantMeasurement& variant_measurement) {
  return mlx_perf::parse_json(
      mlx_perf::write_json(mlx_perf::build_measurement_document(variant_measurement)));
}

double sum_of_slices(const JsonValue& unit_record) {
  double slice_total = 0.0;
  for (const char* slice_field : {"wait_on_host_us", "wait_on_own_stream_us",
                                  "wait_on_other_stream_us", "gpu_scheduling_us",
                                  "compute_us", "result_latency_us"}) {
    slice_total += unit_record.at(slice_field).as_number();
  }
  return slice_total;
}

// --- positive ---

void should_describe_every_node_with_its_identity_and_timing() {
  VariantMeasurement variant_measurement;
  report_fixture::populate_two_stream_run(variant_measurement);
  variant_measurement.variant_name = "eager";
  const JsonValue document = written_and_parsed(variant_measurement);
  EXPECT_TRUE(document.at("schema_version").as_number() == 1);
  EXPECT_TRUE(document.at("granularity").as_string() == "per_node");
  EXPECT_TRUE(document.at("variant_name").as_string() == "eager");
  EXPECT_TRUE(!document.at("mlx_version").as_string().empty());
  const auto& node_records = document.at("nodes").as_array();
  EXPECT_TRUE(node_records.size() == 2);
  const JsonValue& producer = node_records[0];
  EXPECT_TRUE(producer.at("op_name").as_string() == "MatMul");
  EXPECT_TRUE(producer.at("kernel_name").is_null());
  EXPECT_TRUE(producer.at("stream_index").as_number() == 1);
  EXPECT_TRUE(producer.at("shape").as_array().size() == 2);
  EXPECT_TRUE(producer.at("gpu_started_us").as_number() == 15);
  EXPECT_TRUE(producer.at("gpu_finished_us").as_number() == 115);
  EXPECT_TRUE(producer.at("is_compute_gpu_measured").as_bool());
  for (const JsonValue& node_record : node_records) {
    EXPECT_TRUE(std::abs(sum_of_slices(node_record) - node_record.at("wall_us").as_number()) <
                SLICE_SUM_TOLERANCE_MICROSECONDS);
  }
  EXPECT_TRUE(node_records[1].at("classification").as_string() == "wait_on_other_stream");
  EXPECT_TRUE(document.at("stages").as_array().empty());
}

void should_measure_the_gpu_gap_between_consecutive_nodes_on_one_stream() {
  VariantMeasurement variant_measurement;
  report_fixture::populate_two_stream_run(variant_measurement);
  // Put the join on the producer's stream: its GPU start (118) follows the
  // producer's GPU end (115) on the same queue.
  variant_measurement.timeline_graph.nodes[1].stream_index = 1;
  const JsonValue document = written_and_parsed(variant_measurement);
  const auto& node_records = document.at("nodes").as_array();
  EXPECT_TRUE(node_records[0].at("gpu_idle_before_us").is_null());
  EXPECT_TRUE(node_records[1].at("gpu_idle_before_us").as_number() == 3);
}

void should_carry_job_samples_build_times_and_buffer_count() {
  VariantMeasurement variant_measurement;
  report_fixture::populate_two_stream_run(variant_measurement);
  variant_measurement.job_totals.job_wall_sample_microseconds = {90, 80, 120};
  variant_measurement.graph_build_passes = {{"warm_up", {4.0}}, {"job_wall", {2.0, 3.0}}};
  variant_measurement.gpu_timeline.observed_command_buffer_count = 2;
  const JsonValue job_record = written_and_parsed(variant_measurement).at("job");
  EXPECT_TRUE(job_record.at("job_wall_us").at("median").as_number() == 90);
  EXPECT_TRUE(job_record.at("job_wall_us").at("samples").as_array().size() == 3);
  EXPECT_TRUE(job_record.at("command_buffers").at("timeline_pass_observed").as_number() == 2);
  EXPECT_TRUE(!job_record.at("command_buffers").at("is_lower_bound").as_bool());
  const auto& build_records = job_record.at("graph_build").as_array();
  EXPECT_TRUE(build_records.size() == 2);
  EXPECT_TRUE(build_records[1].at("pass").as_string() == "job_wall");
  EXPECT_TRUE(build_records[1].at("total_us").as_number() == 5);
  EXPECT_TRUE(job_record.at("executed_node_count").as_number() == 2);
}

void should_join_each_node_to_its_serial_measurement() {
  VariantMeasurement variant_measurement;
  report_fixture::populate_two_stream_run(variant_measurement);
  const auto& timeline_nodes = variant_measurement.timeline_graph.nodes;
  variant_measurement.serial_telemetry.attributed_nodes = {
      {&timeline_nodes[0], 140.0, 4096, 64}, {&timeline_nodes[1], 9.0, 0, 64}};
  const JsonValue document = written_and_parsed(variant_measurement);
  const auto& node_records = document.at("nodes").as_array();
  EXPECT_TRUE(node_records[0].at("serial_duration_us").as_number() == 140);
  EXPECT_TRUE(node_records[0].at("memory_added_bytes").as_number() == 4096);
}

void should_name_every_classification_in_the_reports_vocabulary() {
  using mlx_perf::NodeClass;
  const std::pair<NodeClass, const char*> class_to_name[] = {
      {NodeClass::compute, "compute"},
      {NodeClass::wait_on_cpu, "wait_on_host"},
      {NodeClass::wait_on_gpu, "wait_on_own_stream"},
      {NodeClass::wait_on_dependency, "wait_on_other_stream"},
      {NodeClass::scheduler_delay, "gpu_scheduling"}};
  for (const auto& [node_class, expected_name] : class_to_name) {
    VariantMeasurement variant_measurement;
    report_fixture::populate_two_stream_run(variant_measurement);
    variant_measurement.gpu_timeline.nodes[0].breakdown.classification = node_class;
    const JsonValue document = written_and_parsed(variant_measurement);
    EXPECT_TRUE(document.at("nodes").as_array()[0].at("classification").as_string() ==
                expected_name);
  }
}

// --- edge / boundary ---

void should_null_the_gpu_window_of_a_node_that_has_none() {
  VariantMeasurement variant_measurement;
  report_fixture::populate_two_stream_run(variant_measurement);
  variant_measurement.gpu_timeline.nodes[0].timestamps.has_gpu_window = false;
  variant_measurement.gpu_timeline.nodes[0].breakdown.is_compute_measured = false;
  const JsonValue document = written_and_parsed(variant_measurement);
  const auto& node_records = document.at("nodes").as_array();
  EXPECT_TRUE(node_records[0].at("gpu_started_us").is_null());
  EXPECT_TRUE(node_records[0].at("gpu_finished_us").is_null());
  EXPECT_TRUE(!node_records[0].at("is_compute_gpu_measured").as_bool());
}

void should_time_stage_tasks_and_null_member_node_timing() {
  VariantMeasurement variant_measurement;
  report_fixture::populate_two_stage_run(variant_measurement);
  variant_measurement.gpu_timeline.is_command_buffer_count_lower_bound = true;
  const JsonValue document = written_and_parsed(variant_measurement);
  EXPECT_TRUE(document.at("granularity").as_string() == "per_stage");
  const auto& stage_records = document.at("stages").as_array();
  EXPECT_TRUE(stage_records.size() == 2);
  EXPECT_TRUE(stage_records[1].at("stage_name").as_string() == "layer 1");
  EXPECT_TRUE(stage_records[1].at("cross_stream_input_stage_positions").as_array().size() == 1);
  EXPECT_TRUE(stage_records[0].at("compute_us").as_number() > 0);
  const JsonValue& member = document.at("nodes").as_array()[0];
  EXPECT_TRUE(member.at("compute_us").is_null());
  EXPECT_TRUE(member.at("gpu_idle_before_us").is_null());
  EXPECT_TRUE(member.at("stage_position").as_number() == 0);
  // Stage members carry exactly the keys a per-node record does.
  VariantMeasurement node_timed_measurement;
  report_fixture::populate_two_stream_run(node_timed_measurement);
  const JsonValue node_timed_document = written_and_parsed(node_timed_measurement);
  const auto& node_timed_member = node_timed_document.at("nodes").as_array()[0].as_object();
  EXPECT_TRUE(member.as_object().size() == node_timed_member.size());
  for (const auto& node_field : node_timed_member) {
    EXPECT_TRUE(member.find(node_field.first) != nullptr);
  }
  EXPECT_TRUE(document.at("job").at("command_buffers").at("is_lower_bound").as_bool());
  // Streams count their nodes, whichever unit timed them.
  for (const JsonValue& stream_record : document.at("streams").as_array()) {
    EXPECT_TRUE(stream_record.find("node_count") != nullptr);
    EXPECT_TRUE(stream_record.find("timed_unit_count") == nullptr);
  }
}

void should_describe_an_empty_measurement_as_empty_lists() {
  const JsonValue document = written_and_parsed(VariantMeasurement{});
  EXPECT_TRUE(document.at("nodes").as_array().empty());
  EXPECT_TRUE(document.at("stages").as_array().empty());
  EXPECT_TRUE(document.at("streams").as_array().empty());
  EXPECT_TRUE(document.at("job").at("job_wall_us").at("samples").as_array().empty());
}

// --- negative ---

void should_null_serial_fields_when_the_serial_node_does_not_match() {
  VariantMeasurement variant_measurement;
  report_fixture::populate_two_stream_run(variant_measurement);
  // A serial graph whose first node ran a different op: never borrow it.
  mlx_perf::GraphNode other_node = report_fixture::make_graph_node("Z", "Sin", 1);
  variant_measurement.serial_graph.nodes = {other_node};
  variant_measurement.serial_telemetry.attributed_nodes = {
      {&variant_measurement.serial_graph.nodes[0], 140.0, 4096, 64}};
  const JsonValue document = written_and_parsed(variant_measurement);
  const auto& node_records = document.at("nodes").as_array();
  EXPECT_TRUE(node_records[0].at("serial_duration_us").is_null());
  EXPECT_TRUE(node_records[0].at("memory_added_bytes").is_null());
  EXPECT_TRUE(node_records[1].at("serial_duration_us").is_null());
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_describe_every_node_with_its_identity_and_timing),
      TEST_CASE(should_measure_the_gpu_gap_between_consecutive_nodes_on_one_stream),
      TEST_CASE(should_carry_job_samples_build_times_and_buffer_count),
      TEST_CASE(should_join_each_node_to_its_serial_measurement),
      TEST_CASE(should_name_every_classification_in_the_reports_vocabulary),
      TEST_CASE(should_null_the_gpu_window_of_a_node_that_has_none),
      TEST_CASE(should_time_stage_tasks_and_null_member_node_timing),
      TEST_CASE(should_describe_an_empty_measurement_as_empty_lists),
      TEST_CASE(should_null_serial_fields_when_the_serial_node_does_not_match),
  });
}
