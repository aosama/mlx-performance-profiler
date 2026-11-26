// Integration tests for measurement_file on the Metal GPU: real jobs are
// measured per node and per stage, written, and read back as a reader in
// another language would read them.

#include "json_value.h"
#include "measurement_file.h"
#include "measurement_session.h"
#include "test_harness.h"

#include "mlx/mlx.h"

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using namespace mlx_perf;
using mlx::core::array;
using mlx::core::Stream;

// ones → matmul → exp on one stream, with an optional stage cut after the
// matmul ("matmul" stage, then "exp").
class SmallStagedJobFactory final : public GraphFactory {
 public:
  explicit SmallStagedJobFactory(const Stream& job_stream) : job_stream_(job_stream) {}
  std::vector<array> build_fresh_graph() const override {
    return build_fresh_staged_graph().output_arrays;
  }
  StagedGraph build_fresh_staged_graph() const override {
    const array job_input = mlx::core::ones({64, 64}, mlx::core::float32, job_stream_);
    const array product = mlx::core::matmul(job_input, job_input, job_stream_);
    return {{mlx::core::exp(product, job_stream_)}, {{"matmul", {product}}}, "exp"};
  }

 private:
  Stream job_stream_;
};

VariantMeasurement measure_small_job(GranularityChoice granularity_choice) {
  const Stream job_stream = mlx::core::new_stream(mlx::core::Device::gpu);
  const std::unordered_map<int, Stream> stream_index_to_stream = {
      {job_stream.index, job_stream}};
  return measure_variant("small", SmallStagedJobFactory(job_stream), stream_index_to_stream,
                         nullptr, {granularity_choice});
}

// --- positive ---

void should_write_a_per_node_measurement_that_reads_back() {
  const VariantMeasurement variant_measurement = measure_small_job(GranularityChoice::per_node);
  const std::string measurement_path =
      write_measurement_file(variant_measurement, ".", "measurement_file_per_node");
  EXPECT_TRUE(measurement_path == "./measurement_file_per_node.json");
  const JsonValue document = parse_json_file(measurement_path);
  EXPECT_TRUE(document.at("granularity").as_string() == "per_node");
  const auto& node_records = document.at("nodes").as_array();
  EXPECT_TRUE(node_records.size() == variant_measurement.gpu_timeline.nodes.size());
  for (const JsonValue& node_record : node_records) {
    EXPECT_TRUE(node_record.at("compute_us").is_number());
    EXPECT_TRUE(node_record.at("serial_duration_us").is_number());
  }
}

void should_write_a_per_stage_measurement_with_stage_timing_only() {
  const VariantMeasurement variant_measurement = measure_small_job(GranularityChoice::per_stage);
  const std::string measurement_path =
      write_measurement_file(variant_measurement, ".", "measurement_file_per_stage");
  const JsonValue document = parse_json_file(measurement_path);
  EXPECT_TRUE(document.at("granularity").as_string() == "per_stage");
  const auto& stage_records = document.at("stages").as_array();
  EXPECT_TRUE(stage_records.size() == 2);
  for (const JsonValue& stage_record : stage_records) {
    EXPECT_TRUE(stage_record.at("compute_us").is_number());
    EXPECT_TRUE(stage_record.at("is_compute_gpu_measured").as_bool());
  }
  if (stage_records.size() == 2) {
    EXPECT_TRUE(stage_records[1].at("gpu_idle_before_us").is_number());
  }
  for (const JsonValue& node_record : document.at("nodes").as_array()) {
    EXPECT_TRUE(node_record.at("compute_us").is_null());
    EXPECT_TRUE(node_record.at("stage_position").is_number());
  }
  EXPECT_TRUE(document.at("job").at("command_buffers").at("is_lower_bound").as_bool());
}

// --- edge / boundary ---

void should_replace_an_earlier_file_of_the_same_name() {
  {
    std::ofstream stale_file("measurement_file_replaced.json");
    stale_file << "not json";
  }
  const VariantMeasurement variant_measurement = measure_small_job(GranularityChoice::per_node);
  const std::string measurement_path =
      write_measurement_file(variant_measurement, ".", "measurement_file_replaced");
  EXPECT_TRUE(parse_json_file(measurement_path).at("schema_version").as_number() == 1);
}

// --- negative ---

void should_return_an_empty_path_when_the_directory_is_missing() {
  const VariantMeasurement empty_measurement;
  EXPECT_TRUE(write_measurement_file(empty_measurement, "no/such/directory", "x").empty());
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_write_a_per_node_measurement_that_reads_back),
      TEST_CASE(should_write_a_per_stage_measurement_with_stage_timing_only),
      TEST_CASE(should_replace_an_earlier_file_of_the_same_name),
      TEST_CASE(should_return_an_empty_path_when_the_directory_is_missing),
  });
}
