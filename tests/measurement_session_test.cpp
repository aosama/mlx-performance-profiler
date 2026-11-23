// Integration tests for measurement_session on the Metal GPU: the four
// passes, the job-wall sampling, and pointer safety of the move-only result.

#include "measurement_session.h"
#include "test_harness.h"

#include "mlx/mlx.h"

#include <utility>
#include <vector>

namespace {

using namespace mlx_plan;
using mlx::core::array;
using mlx::core::Stream;

class SmallJobFactory final : public GraphFactory {
 public:
  explicit SmallJobFactory(const Stream& job_stream) : job_stream_(job_stream) {}
  std::vector<array> build_fresh_graph() const override {
    const array job_input = mlx::core::ones({64, 64}, mlx::core::float32, job_stream_);
    return {mlx::core::exp(mlx::core::matmul(job_input, job_input, job_stream_),
                           job_stream_)};
  }

 private:
  Stream job_stream_;
};

// A factory whose "job" is already materialized: nothing left to dispatch.
class MaterializedJobFactory final : public GraphFactory {
 public:
  std::vector<array> build_fresh_graph() const override {
    array materialized_output = mlx::core::ones({4});
    materialized_output.eval();
    return {materialized_output};
  }
};

// The same small job, but naming one stage cut (after the matmul).
class StagedSmallJobFactory final : public GraphFactory {
 public:
  explicit StagedSmallJobFactory(const Stream& job_stream) : job_stream_(job_stream) {}
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

std::unordered_map<int, Stream> map_single_stream(const Stream& job_stream) {
  return {{job_stream.index, job_stream}};
}

// --- positive ---

void should_measure_all_passes_with_an_ordered_job_wall_range() {
  const Stream job_stream = mlx::core::new_stream(mlx::core::Device::gpu);
  const VariantMeasurement variant_measurement = measure_variant(
      "small", SmallJobFactory(job_stream), map_single_stream(job_stream));
  const JobTotals& job_totals = variant_measurement.job_totals;
  EXPECT_TRUE(job_totals.job_wall_sample_count == 7);
  EXPECT_TRUE(job_totals.job_wall_min_microseconds <= job_totals.job_wall_microseconds);
  EXPECT_TRUE(job_totals.job_wall_microseconds <= job_totals.job_wall_max_microseconds);
  EXPECT_TRUE(job_totals.sum_of_stages_microseconds > 0);
  // Both passes must cover every primitive their (identical) graphs hold.
  const std::size_t primitive_count =
      variant_measurement.timeline_graph.topological.size();
  EXPECT_TRUE(primitive_count > 0);
  EXPECT_TRUE(variant_measurement.gpu_timeline.nodes.size() == primitive_count);
  EXPECT_TRUE(variant_measurement.serial_telemetry.attributed_nodes.size() ==
              primitive_count);
}

// Performance test (asserts a threshold): timing each node alone, synchronized,
// can never beat running the whole job overlapped. If it does, a pass is
// mismeasuring — this is the decomposition invariant from PHASED_PLAN §6.
void should_never_make_sum_of_stages_smaller_than_job_duration() {
  const Stream job_stream = mlx::core::new_stream(mlx::core::Device::gpu);
  const VariantMeasurement variant_measurement = measure_variant(
      "invariant", SmallJobFactory(job_stream), map_single_stream(job_stream));
  EXPECT_TRUE(variant_measurement.job_totals.sum_of_stages_microseconds >=
              variant_measurement.job_totals.job_wall_min_microseconds);
}

void should_time_stages_when_asked_and_cover_every_primitive() {
  const Stream job_stream = mlx::core::new_stream(mlx::core::Device::gpu);
  const VariantMeasurement variant_measurement = measure_variant(
      "staged", StagedSmallJobFactory(job_stream), map_single_stream(job_stream), nullptr,
      {GranularityChoice::per_stage});
  const GpuTimeline& gpu_timeline = variant_measurement.gpu_timeline;
  EXPECT_TRUE(gpu_timeline.granularity == TimelineGranularity::per_stage);
  EXPECT_TRUE(gpu_timeline.stages.size() == 2);
  if (gpu_timeline.stages.size() == 2) {
    EXPECT_TRUE(gpu_timeline.stages[0].stage_name == "matmul");
    EXPECT_TRUE(gpu_timeline.stages[1].stage_name == "exp");
  }
  EXPECT_TRUE(gpu_timeline.nodes.size() ==
              variant_measurement.timeline_graph.topological.size());
}

// --- edge / boundary ---

// Automatic keeps per-node detail for a graph this small, even with cuts.
void should_keep_per_node_automatically_for_a_small_staged_job() {
  const Stream job_stream = mlx::core::new_stream(mlx::core::Device::gpu);
  const VariantMeasurement variant_measurement = measure_variant(
      "small staged", StagedSmallJobFactory(job_stream), map_single_stream(job_stream));
  EXPECT_TRUE(variant_measurement.gpu_timeline.granularity == TimelineGranularity::per_node);
  EXPECT_TRUE(variant_measurement.gpu_timeline.stages.empty());
}

// Forcing stages on a job with no cuts times the whole job as one stage.
void should_time_a_job_without_cuts_as_one_whole_job_stage() {
  const Stream job_stream = mlx::core::new_stream(mlx::core::Device::gpu);
  const VariantMeasurement variant_measurement = measure_variant(
      "whole", SmallJobFactory(job_stream), map_single_stream(job_stream), nullptr,
      {GranularityChoice::per_stage});
  const GpuTimeline& gpu_timeline = variant_measurement.gpu_timeline;
  EXPECT_TRUE(gpu_timeline.stages.size() == 1);
  if (!gpu_timeline.stages.empty()) {
    EXPECT_TRUE(gpu_timeline.stages.front().stage_name == "whole job");
  }
}

void should_keep_node_pointers_valid_after_the_result_is_moved() {
  const Stream job_stream = mlx::core::new_stream(mlx::core::Device::gpu);
  std::vector<VariantMeasurement> measured_variants;
  measured_variants.push_back(measure_variant(
      "first", SmallJobFactory(job_stream), map_single_stream(job_stream)));
  // Force a reallocation: the first measurement is moved to a new buffer.
  measured_variants.reserve(16);
  const VariantMeasurement& moved_measurement = measured_variants.front();
  const GraphNode* first_graph_node = &moved_measurement.timeline_graph.nodes.front();
  const GraphNode* past_last_graph_node =
      first_graph_node + moved_measurement.timeline_graph.nodes.size();
  for (const NodeTimeline& node_timeline : moved_measurement.gpu_timeline.nodes) {
    EXPECT_TRUE(node_timeline.node >= first_graph_node &&
                node_timeline.node < past_last_graph_node);
  }
}

// --- negative ---

void should_return_empty_measurement_for_an_already_materialized_job() {
  const VariantMeasurement variant_measurement =
      measure_variant("materialized", MaterializedJobFactory(), {});
  EXPECT_TRUE(variant_measurement.gpu_timeline.nodes.empty());
  EXPECT_TRUE(variant_measurement.serial_telemetry.attributed_nodes.empty());
  EXPECT_TRUE(variant_measurement.job_totals.job_wall_sample_count == 7);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_measure_all_passes_with_an_ordered_job_wall_range),
      TEST_CASE(should_never_make_sum_of_stages_smaller_than_job_duration),
      TEST_CASE(should_time_stages_when_asked_and_cover_every_primitive),
      TEST_CASE(should_keep_per_node_automatically_for_a_small_staged_job),
      TEST_CASE(should_time_a_job_without_cuts_as_one_whole_job_stage),
      TEST_CASE(should_keep_node_pointers_valid_after_the_result_is_moved),
      TEST_CASE(should_return_empty_measurement_for_an_already_materialized_job),
  });
}
