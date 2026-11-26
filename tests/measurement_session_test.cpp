// Integration tests for measurement_session on the Metal GPU: the four
// passes, the job-wall sampling, and pointer safety of the move-only result.

#include "measurement_session.h"
#include "test_harness.h"

#include "mlx/mlx.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace mlx_perf;
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

void should_keep_every_job_wall_sample_and_time_every_graph_build() {
  const Stream job_stream = mlx::core::new_stream(mlx::core::Device::gpu);
  const VariantMeasurement variant_measurement = measure_variant(
      "samples", SmallJobFactory(job_stream), map_single_stream(job_stream));
  const JobTotals& job_totals = variant_measurement.job_totals;
  EXPECT_TRUE(job_totals.job_wall_sample_microseconds.size() == 7);
  for (const double sample_microseconds : job_totals.job_wall_sample_microseconds) {
    EXPECT_TRUE(sample_microseconds >= job_totals.job_wall_min_microseconds);
    EXPECT_TRUE(sample_microseconds <= job_totals.job_wall_max_microseconds);
  }
  // warm-up 1, job wall 7, timeline 1, serial 1: one entry per pass, in order.
  const auto& build_passes = variant_measurement.graph_build_passes;
  EXPECT_TRUE(build_passes.size() == 4);
  if (build_passes.size() == 4) {
    EXPECT_TRUE(build_passes[0].pass_name == "warm_up");
    EXPECT_TRUE(build_passes[1].pass_name == "job_wall");
    EXPECT_TRUE(build_passes[1].call_microseconds.size() == 7);
    EXPECT_TRUE(build_passes[2].pass_name == "timeline");
    EXPECT_TRUE(build_passes[3].pass_name == "serial");
    EXPECT_TRUE(build_passes[3].call_microseconds.size() == 1);
  }
}

void should_return_evaluated_outputs_from_the_warm_up() {
  const Stream job_stream = mlx::core::new_stream(mlx::core::Device::gpu);
  const std::vector<array> warm_up_outputs = run_warm_up(SmallJobFactory(job_stream));
  EXPECT_TRUE(warm_up_outputs.size() == 1);
  EXPECT_TRUE(!warm_up_outputs.empty() && warm_up_outputs[0].is_available());
}

void should_record_median_range_and_samples_in_measured_order() {
  JobTotals job_totals;
  record_job_wall_samples({30, 10, 20, 40}, job_totals);
  EXPECT_TRUE(job_totals.job_wall_microseconds == 25);  // mean of the two middle samples
  EXPECT_TRUE(job_totals.job_wall_min_microseconds == 10);
  EXPECT_TRUE(job_totals.job_wall_max_microseconds == 40);
  EXPECT_TRUE(job_totals.job_wall_sample_count == 4);
  EXPECT_TRUE((job_totals.job_wall_sample_microseconds == std::vector<double>{30, 10, 20, 40}));
}

void should_measure_the_whole_jobs_gpu_span_as_one_unit() {
  const Stream job_stream = mlx::core::new_stream(mlx::core::Device::gpu);
  EXPECT_TRUE(measure_gpu_span_once(SmallJobFactory(job_stream),
                                    map_single_stream(job_stream)) > 0);
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

// A CPU-only job has no command buffer, so no GPU span: zero, not an error.
void should_measure_no_gpu_span_for_a_cpu_only_job() {
  const Stream cpu_stream = mlx::core::new_stream(mlx::core::Device::cpu);
  EXPECT_TRUE(measure_gpu_span_once(SmallJobFactory(cpu_stream),
                                    map_single_stream(cpu_stream)) == 0);
}

void should_refuse_a_gpu_span_on_an_unknown_stream() {
  const Stream job_stream = mlx::core::new_stream(mlx::core::Device::gpu);
  bool has_refused = false;
  try {
    measure_gpu_span_once(SmallJobFactory(job_stream), {});
  } catch (const std::out_of_range&) {
    has_refused = true;
  }
  EXPECT_TRUE(has_refused);
}

void should_refuse_to_record_no_job_wall_samples() {
  JobTotals job_totals;
  bool has_refused = false;
  try {
    record_job_wall_samples({}, job_totals);
  } catch (const std::invalid_argument&) {
    has_refused = true;
  }
  EXPECT_TRUE(has_refused);
}

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
      TEST_CASE(should_keep_every_job_wall_sample_and_time_every_graph_build),
      TEST_CASE(should_return_evaluated_outputs_from_the_warm_up),
      TEST_CASE(should_record_median_range_and_samples_in_measured_order),
      TEST_CASE(should_measure_the_whole_jobs_gpu_span_as_one_unit),
      TEST_CASE(should_never_make_sum_of_stages_smaller_than_job_duration),
      TEST_CASE(should_time_stages_when_asked_and_cover_every_primitive),
      TEST_CASE(should_keep_per_node_automatically_for_a_small_staged_job),
      TEST_CASE(should_time_a_job_without_cuts_as_one_whole_job_stage),
      TEST_CASE(should_keep_node_pointers_valid_after_the_result_is_moved),
      TEST_CASE(should_measure_no_gpu_span_for_a_cpu_only_job),
      TEST_CASE(should_refuse_a_gpu_span_on_an_unknown_stream),
      TEST_CASE(should_refuse_to_record_no_job_wall_samples),
      TEST_CASE(should_return_empty_measurement_for_an_already_materialized_job),
  });
}
