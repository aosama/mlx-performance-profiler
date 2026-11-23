#include "measurement_session.h"

#include "progress_reporter.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace mlx_plan {
namespace {

constexpr int JOB_WALL_SAMPLE_COUNT = 7;

// Each sample evaluates its own fresh graph; the median resists the
// occasional slow run without hiding it (min/max are reported too).
void sample_job_wall(const GraphFactory& graph_factory, JobTotals& job_totals) {
  std::vector<double> job_wall_samples;
  ProgressReporter progress("job wall samples", JOB_WALL_SAMPLE_COUNT);
  for (int sample_index = 0; sample_index < JOB_WALL_SAMPLE_COUNT; ++sample_index) {
    job_wall_samples.push_back(run_pipeline_pass(graph_factory.build_fresh_graph()));
    progress.update(sample_index + 1);
  }
  progress.finish(JOB_WALL_SAMPLE_COUNT);
  std::sort(job_wall_samples.begin(), job_wall_samples.end());
  job_totals.job_wall_microseconds = job_wall_samples[JOB_WALL_SAMPLE_COUNT / 2];
  job_totals.job_wall_min_microseconds = job_wall_samples.front();
  job_totals.job_wall_max_microseconds = job_wall_samples.back();
  job_totals.job_wall_sample_count = JOB_WALL_SAMPLE_COUNT;
}

CapturedGraph capture_fresh_graph(const GraphFactory& graph_factory) {
  CapturedGraph fresh_graph = capture_graph(graph_factory.build_fresh_graph());
  std::printf(
      "    captured %zu nodes (%zu primitives), %zu edges, %d streams\n",
      fresh_graph.nodes.size(), fresh_graph.topological.size(),
      fresh_graph.edges.size(), fresh_graph.stream_count);
  return fresh_graph;
}

void announce_pass(MeasurementObserver* measurement_observer,
                   const std::string& variant_name,
                   const std::string& pass_description) {
  std::printf("[%s] %s...\n", variant_name.c_str(), pass_description.c_str());
  if (measurement_observer != nullptr) {
    measurement_observer->on_pass_started(variant_name, pass_description);
  }
}

} // namespace

VariantMeasurement measure_variant(
    const std::string& variant_name,
    const GraphFactory& graph_factory,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream,
    MeasurementObserver* measurement_observer) {
  VariantMeasurement variant_measurement;
  variant_measurement.variant_name = variant_name;

  // Discarded: the first eval pays Metal kernel compilation (and, for a
  // compiled variant, tracing and fused-kernel JIT), which is not job cost.
  announce_pass(measurement_observer, variant_name,
                "pass 0/3: warm-up (compile/JIT; discarded)");
  run_pipeline_pass(graph_factory.build_fresh_graph());

  announce_pass(measurement_observer, variant_name,
                "pass 1/3: pipeline pass (job wall, median of " +
                    std::to_string(JOB_WALL_SAMPLE_COUNT) + ")");
  sample_job_wall(graph_factory, variant_measurement.job_totals);

  announce_pass(measurement_observer, variant_name,
                "pass 2/3: timeline pass (per-node, overlap kept)");
  variant_measurement.timeline_graph = capture_fresh_graph(graph_factory);
  variant_measurement.gpu_timeline = capture_gpu_timeline(
      variant_measurement.timeline_graph, stream_index_to_stream);

  announce_pass(measurement_observer, variant_name,
                "pass 3/3: serial pass (sum of stages)");
  variant_measurement.serial_graph = capture_fresh_graph(graph_factory);
  variant_measurement.serial_telemetry = run_attributed_pass(
      variant_measurement.serial_graph, stream_index_to_stream);
  variant_measurement.job_totals.sum_of_stages_microseconds =
      variant_measurement.serial_telemetry.attributed_total_microseconds;
  return variant_measurement;
}

} // namespace mlx_plan
