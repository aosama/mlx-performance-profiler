#include "measurement_session.h"

#include "console_output.h"
#include "progress_reporter.h"
#include "sample_statistics.h"
#include "stage_timeline.h"
#include "metal_dispatch_recorder.h"

#include <cstdio>
#include <string>
#include <vector>

namespace mlx_perf {
namespace {

// Each sample evaluates its own fresh graph; the median resists the
// occasional slow run without hiding it (min/max are reported too).
void sample_job_wall(const GraphFactory& graph_factory, JobTotals& job_totals) {
  std::vector<double> job_wall_samples;
  ProgressReporter progress("job wall samples", JOB_WALL_SAMPLE_COUNT);
  for (int sample_index = 0; sample_index < JOB_WALL_SAMPLE_COUNT; ++sample_index) {
    job_wall_samples.push_back(sample_job_wall_once(graph_factory).job_wall_microseconds);
    progress.update(sample_index + 1);
  }
  progress.finish(JOB_WALL_SAMPLE_COUNT);
  record_job_wall_samples(job_wall_samples, job_totals);
}

CapturedGraph capture_fresh_graph(const GraphFactory& graph_factory) {
  CapturedGraph fresh_graph = capture_graph(graph_factory.build_fresh_graph());
  print_progress_line(
      "    captured %zu nodes (%zu primitives), %zu edges, %d streams\n",
      fresh_graph.nodes.size(), fresh_graph.topological.size(),
      fresh_graph.edges.size(), fresh_graph.stream_count);
  return fresh_graph;
}

void announce_pass(MeasurementObserver* measurement_observer,
                   const std::string& variant_name,
                   const std::string& pass_description) {
  print_progress_line("[%s] %s...\n", variant_name.c_str(), pass_description.c_str());

  if (measurement_observer != nullptr) {
    measurement_observer->on_pass_started(variant_name, pass_description);
  }
}

} // namespace

std::vector<mlx::core::array> run_warm_up(const GraphFactory& graph_factory) {
  // Before the warm-up compiles the job's kernels, so every pipeline it
  // compiles is named; kernels compiled earlier in the process stay unnamed.
  install_dispatch_recorder();
  std::vector<mlx::core::array> warm_up_outputs = graph_factory.build_fresh_graph();
  // The pass takes its own copy; copies share each array's data, so these
  // outputs are evaluated when it returns.
  run_pipeline_pass(warm_up_outputs);
  return warm_up_outputs;
}

PipelinePassTiming sample_job_wall_once(const GraphFactory& graph_factory) {
  return run_pipeline_pass(graph_factory.build_fresh_graph());
}

double measure_gpu_span_once(
    const GraphFactory& graph_factory,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream) {
  const CapturedGraph whole_job_graph = capture_graph(graph_factory.build_fresh_graph());
  // No stage cuts: the whole job is one stage, one task per stream.
  const GpuTimeline whole_job_timeline =
      capture_stage_timeline(whole_job_graph, stream_index_to_stream, {}, "whole job");
  return whole_job_timeline.measured_gpu_compute_microseconds;
}

void record_job_wall_samples(const std::vector<double>& job_wall_samples,
                             JobTotals& job_totals) {
  const SampleSummary job_wall_summary = summarize_samples(job_wall_samples);
  job_totals.job_wall_sample_microseconds = job_wall_samples;
  job_totals.job_wall_microseconds = job_wall_summary.median;
  job_totals.job_wall_min_microseconds = job_wall_summary.minimum;
  job_totals.job_wall_max_microseconds = job_wall_summary.maximum;
  job_totals.job_wall_sample_count = job_wall_summary.sample_count;
}

// Stage cuts are resolved to positions right after capture and the boundary
// handles dropped, so the pass holds no array the plain job would not.
TimelineCapture capture_timeline(
    const GraphFactory& graph_factory,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream,
    const TimelineOptions& timeline_options) {
  TimelineCapture timeline_capture;
  StagedGraph staged_graph = graph_factory.build_fresh_staged_graph();
  timeline_capture.timeline_graph = capture_graph(staged_graph.output_arrays);
  const CapturedGraph& timeline_graph = timeline_capture.timeline_graph;
  const TimelineGranularity granularity = choose_granularity(
      timeline_options.granularity_choice,
      static_cast<int>(staged_graph.stage_boundaries.size()),
      static_cast<int>(timeline_graph.topological.size()));
  const std::vector<StageCut> stage_cuts =
      resolve_stage_cuts(timeline_graph, staged_graph.stage_boundaries);
  const std::string final_stage_name = stage_cuts.empty()
      ? std::string("whole job")
      : staged_graph.final_stage_name;
  staged_graph = {};
  print_progress_line("    captured %zu nodes (%zu primitives), %d streams; timing %s"
              " (%zu job-supplied stage cuts)\n",
              timeline_graph.nodes.size(), timeline_graph.topological.size(),
              timeline_graph.stream_count, granularity_name(granularity).data(),
              stage_cuts.size());
  timeline_capture.gpu_timeline =
      granularity == TimelineGranularity::per_stage
      ? capture_stage_timeline(timeline_graph, stream_index_to_stream, stage_cuts,
                               final_stage_name)
      : capture_gpu_timeline(timeline_graph, stream_index_to_stream);
  return timeline_capture;
}

void run_serial_pass(const GraphFactory& graph_factory,
                     const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream,
                     VariantMeasurement& variant_measurement) {
  variant_measurement.serial_graph = capture_fresh_graph(graph_factory);
  variant_measurement.serial_telemetry = run_attributed_pass(
      variant_measurement.serial_graph, stream_index_to_stream);
  variant_measurement.job_totals.sum_of_stages_microseconds =
      variant_measurement.serial_telemetry.attributed_total_microseconds;
}

VariantMeasurement measure_variant(
    const std::string& variant_name,
    const GraphFactory& graph_factory,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream,
    MeasurementObserver* measurement_observer,
    const TimelineOptions& timeline_options) {
  VariantMeasurement variant_measurement;
  variant_measurement.variant_name = variant_name;
  // Every pass builds through the timer, so graph-construction time is
  // recorded per pass without any pass knowing about it.
  TimedGraphFactory timed_factory(graph_factory);

  announce_pass(measurement_observer, variant_name,
                "pass 0/3: warm-up (compile/JIT; discarded)");
  timed_factory.begin_pass("warm_up");
  run_warm_up(timed_factory);

  announce_pass(measurement_observer, variant_name,
                "pass 1/3: pipeline pass (job wall, median of " +
                    std::to_string(JOB_WALL_SAMPLE_COUNT) + ")");
  timed_factory.begin_pass("job_wall");
  sample_job_wall(timed_factory, variant_measurement.job_totals);

  announce_pass(measurement_observer, variant_name,
                "pass 2/3: timeline pass (overlap kept)");
  timed_factory.begin_pass("timeline");
  TimelineCapture timeline_capture =
      capture_timeline(timed_factory, stream_index_to_stream, timeline_options);
  variant_measurement.timeline_graph = std::move(timeline_capture.timeline_graph);
  variant_measurement.gpu_timeline = std::move(timeline_capture.gpu_timeline);

  announce_pass(measurement_observer, variant_name,
                "pass 3/3: serial pass (sum of stages)");
  timed_factory.begin_pass("serial");
  run_serial_pass(timed_factory, stream_index_to_stream, variant_measurement);
  variant_measurement.graph_build_passes = timed_factory.build_passes();
  return variant_measurement;
}

} // namespace mlx_perf
