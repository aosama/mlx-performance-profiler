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

// The build log reads the schedule back: warm-ups A B; 12 uninstrumented
// rounds and 12 GPU-span rounds, each A B then B A alternately; then one
// timeline and one serial pass each, A B.
void should_alternate_the_variants_and_flip_the_order_every_round() {
  PairFixture fixture;
  const LoggingJobFactory exp_job("exp", fixture.job_stream, false, fixture.build_log);
  const LoggingJobFactory sin_job("sin", fixture.job_stream, true, fixture.build_log);
  measure_pair({"exp", exp_job}, {"sin", sin_job}, fixture.stream_index_to_stream);
  const std::vector<std::string>& build_log = fixture.build_log;
  constexpr std::size_t ROUND_BUILDS = 2 * PAIRED_ROUND_COUNT;
  EXPECT_TRUE(build_log.size() == 2 + ROUND_BUILDS + ROUND_BUILDS + 2 + 2);
  if (build_log.size() != 2 + 2 * ROUND_BUILDS + 4) {
    return;
  }
  for (const std::size_t round_start : {std::size_t{2}, 2 + ROUND_BUILDS}) {
    for (int round_index = 0; round_index < PAIRED_ROUND_COUNT; ++round_index) {
      const std::string& first_runner = build_log[round_start + 2 * round_index];
      EXPECT_TRUE(first_runner == (round_index % 2 == 0 ? "exp" : "sin"));
    }
  }
  const std::size_t tail_start = 2 + 2 * ROUND_BUILDS;
  EXPECT_TRUE(build_log[tail_start] == "exp" && build_log[tail_start + 1] == "sin");
  EXPECT_TRUE(build_log[tail_start + 2] == "exp" && build_log[tail_start + 3] == "sin");
  EXPECT_TRUE(count_builds(build_log, "exp") == count_builds(build_log, "sin"));
}

void should_keep_every_rounds_numbers_and_one_illustrative_timeline() {
  PairFixture fixture;
  const LoggingJobFactory exp_job("exp", fixture.job_stream, false, fixture.build_log);
  const LoggingJobFactory sin_job("sin", fixture.job_stream, true, fixture.build_log);
  const PairedMeasurement paired_measurement =
      measure_pair({"exp", exp_job}, {"sin", sin_job}, fixture.stream_index_to_stream);
  for (const VariantRounds* variant_rounds :
       {&paired_measurement.baseline_rounds, &paired_measurement.candidate_rounds}) {
    EXPECT_TRUE(variant_rounds->job_wall_microseconds.size() == PAIRED_ROUND_COUNT);
    EXPECT_TRUE(variant_rounds->host_submission_microseconds.size() == PAIRED_ROUND_COUNT);
    EXPECT_TRUE(variant_rounds->gpu_span_microseconds.size() == PAIRED_ROUND_COUNT);
    for (std::size_t round_index = 0; round_index < PAIRED_ROUND_COUNT; ++round_index) {
      EXPECT_TRUE(variant_rounds->host_submission_microseconds[round_index] <=
                  variant_rounds->job_wall_microseconds[round_index]);
      EXPECT_TRUE(variant_rounds->gpu_span_microseconds[round_index] > 0);
    }
  }
  EXPECT_TRUE(paired_measurement.baseline.variant_name == "exp");
  EXPECT_TRUE(paired_measurement.candidate.gpu_timeline.granularity ==
              TimelineGranularity::per_node);
  EXPECT_TRUE(!paired_measurement.candidate.gpu_timeline.nodes.empty());
  EXPECT_TRUE(paired_measurement.candidate.serial_telemetry.attributed_nodes.size() ==
              paired_measurement.candidate.timeline_graph.topological.size());
  EXPECT_TRUE(paired_measurement.output_agreement.output_count == 1);
}

void should_compare_the_pair_on_job_wall_gpu_span_and_host_submission() {
  PairFixture fixture;
  const LoggingJobFactory exp_job("exp", fixture.job_stream, false, fixture.build_log);
  const LoggingJobFactory sin_job("sin", fixture.job_stream, true, fixture.build_log);
  const PairComparisons pair_comparisons = compare_pair(
      measure_pair({"exp", exp_job}, {"sin", sin_job}, fixture.stream_index_to_stream));
  EXPECT_TRUE(pair_comparisons.job_wall.round_count == PAIRED_ROUND_COUNT);
  EXPECT_TRUE(pair_comparisons.gpu_span.round_count == PAIRED_ROUND_COUNT);
  EXPECT_TRUE(pair_comparisons.host_submission.baseline.median > 0);
  EXPECT_TRUE(pair_comparisons.job_wall.difference_interval_coverage >=
              MINIMUM_INTERVAL_COVERAGE);
  // Memory is judged on the same uninstrumented rounds (#46).
  EXPECT_TRUE(pair_comparisons.peak_mlx_active_added.round_count == PAIRED_ROUND_COUNT);
  EXPECT_TRUE(pair_comparisons.peak_mlx_active_added.baseline.median > 0);
  EXPECT_TRUE(pair_comparisons.system_wired_at_end.baseline.median > 0);
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

// Collects what a live view's status line would show.
class RecordingObserver final : public MeasurementObserver {
 public:
  void on_pass_started(const std::string& variant_name,
                       const std::string& pass_description) override {
    announced_passes.push_back(variant_name + ": " + pass_description);
  }
  std::vector<std::string> announced_passes;
};

void should_tell_an_observer_as_each_phase_begins() {
  PairFixture fixture;
  const LoggingJobFactory exp_job("exp", fixture.job_stream, false, fixture.build_log);
  const LoggingJobFactory sin_job("sin", fixture.job_stream, true, fixture.build_log);
  RecordingObserver recording_observer;
  measure_pair({"exp", exp_job}, {"sin", sin_job}, fixture.stream_index_to_stream, {},
               &recording_observer);
  // warm-up, uninstrumented rounds, GPU-span rounds, timeline, serial.
  EXPECT_TRUE(recording_observer.announced_passes.size() == 5);
  if (!recording_observer.announced_passes.empty()) {
    EXPECT_TRUE(recording_observer.announced_passes.front().rfind("exp vs sin: warm-up", 0) == 0);
  }
}

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
  EXPECT_TRUE(compare_pair(paired_measurement).gpu_span.baseline.median > 0);
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
      TEST_CASE(should_keep_every_rounds_numbers_and_one_illustrative_timeline),
      TEST_CASE(should_compare_the_pair_on_job_wall_gpu_span_and_host_submission),
      TEST_CASE(should_tell_an_observer_as_each_phase_begins),
      TEST_CASE(should_time_both_variants_per_stage_when_asked),
      TEST_CASE(should_find_a_job_compared_with_itself_in_full_agreement),
      TEST_CASE(should_refuse_jobs_with_different_output_shapes_before_timing_anything),
  });
}
