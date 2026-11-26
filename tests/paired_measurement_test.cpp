// Integration tests for paired_measurement on the Metal GPU: the schedule's
// interleaving (read from a shared log of graph builds), the per-round
// numbers, the median timeline, and the refusal of mismatched outputs.

#include "paired_measurement.h"
#include "test_harness.h"

#include "mlx/mlx.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using namespace mlx_perf;
using mlx::core::array;
using mlx::core::Stream;

// Builds ones → matmul → (exp | sin) on one stream and logs its name in the
// shared build log on every call, so the test can read the schedule back.
class LoggingJobFactory final : public GraphFactory {
 public:
  LoggingJobFactory(std::string job_name, const Stream& job_stream, bool uses_sin,
                    std::vector<std::string>& build_log, int output_width = 64)
      : job_name_(std::move(job_name)),
        job_stream_(job_stream),
        uses_sin_(uses_sin),
        build_log_(build_log),
        output_width_(output_width) {}
  std::vector<array> build_fresh_graph() const override {
    build_log_.push_back(job_name_);
    const array job_input =
        mlx::core::ones({64, output_width_}, mlx::core::float32, job_stream_);
    const array product = mlx::core::matmul(
        mlx::core::ones({64, 64}, mlx::core::float32, job_stream_), job_input, job_stream_);
    return {uses_sin_ ? mlx::core::sin(product, job_stream_)
                      : mlx::core::exp(product, job_stream_)};
  }

 private:
  std::string job_name_;
  Stream job_stream_;
  bool uses_sin_;
  std::vector<std::string>& build_log_;
  int output_width_;
};

struct PairFixture {
  Stream job_stream = mlx::core::new_stream(mlx::core::Device::gpu);
  std::unordered_map<int, Stream> stream_index_to_stream{{job_stream.index, job_stream}};
  std::vector<std::string> build_log;
};

int count_builds(const std::vector<std::string>& build_log, const std::string& job_name) {
  int build_count = 0;
  for (const std::string& logged_name : build_log) {
    build_count += logged_name == job_name ? 1 : 0;
  }
  return build_count;
}

// --- positive ---

void should_alternate_the_variants_and_flip_the_order_every_round() {
  PairFixture fixture;
  const LoggingJobFactory exp_job("exp", fixture.job_stream, false, fixture.build_log);
  const LoggingJobFactory sin_job("sin", fixture.job_stream, true, fixture.build_log);
  measure_pair({"exp", exp_job}, {"sin", sin_job}, fixture.stream_index_to_stream);
  // warm-ups A B; job wall A B, B A, … (7 rounds); timeline A B, B A, … (5); serial A B.
  const std::vector<std::string>& build_log = fixture.build_log;
  EXPECT_TRUE(build_log.size() ==
              static_cast<std::size_t>(2 + 2 * JOB_WALL_SAMPLE_COUNT +
                                       2 * PAIRED_TIMELINE_ROUND_COUNT + 2));
  if (build_log.size() >= 6) {
    EXPECT_TRUE(build_log[0] == "exp" && build_log[1] == "sin");
    EXPECT_TRUE(build_log[2] == "exp" && build_log[3] == "sin");  // job-wall round 0
    EXPECT_TRUE(build_log[4] == "sin" && build_log[5] == "exp");  // job-wall round 1
  }
  EXPECT_TRUE(count_builds(build_log, "exp") == count_builds(build_log, "sin"));
}

void should_keep_per_round_numbers_and_the_median_timeline() {
  PairFixture fixture;
  const LoggingJobFactory exp_job("exp", fixture.job_stream, false, fixture.build_log);
  const LoggingJobFactory sin_job("sin", fixture.job_stream, true, fixture.build_log);
  const PairedMeasurement paired_measurement =
      measure_pair({"exp", exp_job}, {"sin", sin_job}, fixture.stream_index_to_stream);
  for (const VariantRounds* variant_rounds :
       {&paired_measurement.baseline_rounds, &paired_measurement.candidate_rounds}) {
    EXPECT_TRUE(variant_rounds->job_wall_microseconds.size() == JOB_WALL_SAMPLE_COUNT);
    EXPECT_TRUE(variant_rounds->gpu_execution_microseconds.size() ==
                PAIRED_TIMELINE_ROUND_COUNT);
    EXPECT_TRUE(variant_rounds->host_submission_microseconds.size() ==
                PAIRED_TIMELINE_ROUND_COUNT);
  }
  // The kept timeline is the round with the median GPU execution.
  const double kept_gpu_execution =
      paired_measurement.candidate.gpu_timeline.measured_gpu_compute_microseconds;
  EXPECT_TRUE(kept_gpu_execution ==
              summarize_samples(paired_measurement.candidate_rounds.gpu_execution_microseconds)
                  .median);
  EXPECT_TRUE(paired_measurement.baseline.variant_name == "exp");
  EXPECT_TRUE(paired_measurement.candidate.serial_telemetry.attributed_nodes.size() ==
              paired_measurement.candidate.timeline_graph.topological.size());
  EXPECT_TRUE(paired_measurement.output_agreement.output_count == 1);
}

void should_compare_the_pair_on_job_wall_gpu_execution_and_host_submission() {
  PairFixture fixture;
  const LoggingJobFactory exp_job("exp", fixture.job_stream, false, fixture.build_log);
  const LoggingJobFactory sin_job("sin", fixture.job_stream, true, fixture.build_log);
  const PairComparisons pair_comparisons = compare_pair(
      measure_pair({"exp", exp_job}, {"sin", sin_job}, fixture.stream_index_to_stream));
  EXPECT_TRUE(pair_comparisons.job_wall.round_count == JOB_WALL_SAMPLE_COUNT);
  EXPECT_TRUE(pair_comparisons.gpu_execution.round_count == PAIRED_TIMELINE_ROUND_COUNT);
  EXPECT_TRUE(pair_comparisons.host_submission.baseline.median > 0);
}

// Two stage cuts' worth of job (matmul, then the unary op) on one stream.
class StagedLoggingJobFactory final : public GraphFactory {
 public:
  StagedLoggingJobFactory(const Stream& job_stream, bool uses_sin)
      : job_stream_(job_stream), uses_sin_(uses_sin) {}
  std::vector<array> build_fresh_graph() const override {
    return build_fresh_staged_graph().output_arrays;
  }
  StagedGraph build_fresh_staged_graph() const override {
    const array job_input = mlx::core::ones({64, 64}, mlx::core::float32, job_stream_);
    const array product = mlx::core::matmul(job_input, job_input, job_stream_);
    const array finished = uses_sin_ ? mlx::core::sin(product, job_stream_)
                                     : mlx::core::cos(product, job_stream_);
    return {{finished}, {{"matmul", {product}}}, "unary"};
  }

 private:
  Stream job_stream_;
  bool uses_sin_;
};

// --- edge / boundary ---

void should_time_both_variants_per_stage_when_asked() {
  PairFixture fixture;
  const StagedLoggingJobFactory sin_job(fixture.job_stream, true);
  const StagedLoggingJobFactory cos_job(fixture.job_stream, false);
  const PairedMeasurement paired_measurement =
      measure_pair({"sin", sin_job}, {"cos", cos_job}, fixture.stream_index_to_stream,
                   {GranularityChoice::per_stage});
  for (const VariantMeasurement* variant : {&paired_measurement.baseline,
                                            &paired_measurement.candidate}) {
    EXPECT_TRUE(variant->gpu_timeline.granularity == TimelineGranularity::per_stage);
    EXPECT_TRUE(variant->gpu_timeline.stages.size() == 2);
  }
  EXPECT_TRUE(compare_pair(paired_measurement).gpu_execution.baseline.median > 0);
}

void should_find_a_job_compared_with_itself_in_full_agreement() {
  PairFixture fixture;
  const LoggingJobFactory exp_job("exp", fixture.job_stream, false, fixture.build_log);
  const PairedMeasurement paired_measurement =
      measure_pair({"first", exp_job}, {"second", exp_job}, fixture.stream_index_to_stream);
  EXPECT_TRUE(paired_measurement.output_agreement.max_absolute_difference == 0.0);
  EXPECT_TRUE(paired_measurement.candidate.variant_name == "second");
}

// --- negative ---

void should_refuse_jobs_with_different_output_shapes_before_timing_anything() {
  PairFixture fixture;
  const LoggingJobFactory square_job("square", fixture.job_stream, false, fixture.build_log);
  const LoggingJobFactory wide_job("wide", fixture.job_stream, false, fixture.build_log, 128);
  bool has_refused = false;
  try {
    measure_pair({"square", square_job}, {"wide", wide_job}, fixture.stream_index_to_stream);
  } catch (const std::invalid_argument& mismatch) {
    has_refused = std::string(mismatch.what()).find("(64,128)") != std::string::npos;
  }
  EXPECT_TRUE(has_refused);
  EXPECT_TRUE(fixture.build_log.size() == 2);  // the two warm-ups only
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_alternate_the_variants_and_flip_the_order_every_round),
      TEST_CASE(should_keep_per_round_numbers_and_the_median_timeline),
      TEST_CASE(should_compare_the_pair_on_job_wall_gpu_execution_and_host_submission),
      TEST_CASE(should_time_both_variants_per_stage_when_asked),
      TEST_CASE(should_find_a_job_compared_with_itself_in_full_agreement),
      TEST_CASE(should_refuse_jobs_with_different_output_shapes_before_timing_anything),
  });
}
