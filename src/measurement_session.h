#pragma once

// Measurement session: one concern only — run the four passes (warm-up,
// job wall, timeline at the chosen granularity, serial) for one job variant
// and keep everything the reports need alive together. The passes are also
// exposed one call at a time, so paired_measurement can interleave two
// variants pass by pass.

#include "gpu_timeline.h"
#include "graph_build_timer.h"
#include "graph_factory.h"
#include "graph_model.h"
#include "job_totals.h"
#include "stage_plan.h"
#include "telemetry.h"

#include "mlx/stream.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_perf {

// Move-only on purpose: gpu_timeline points into timeline_graph.nodes and
// serial_telemetry into serial_graph.nodes. A move keeps those vectors'
// buffers (and so the pointers) intact; a copy would leave them dangling.
struct VariantMeasurement {
  std::string variant_name;
  CapturedGraph timeline_graph;
  GpuTimeline gpu_timeline;
  CapturedGraph serial_graph;
  RunTelemetry serial_telemetry;
  JobTotals job_totals;
  // Time spent inside the job's own graph-building code, per pass
  // ("warm_up", "job_wall", "timeline", "serial"), outside every pass clock.
  // The C API puts its own "stream_discovery" build first.
  std::vector<GraphBuildPass> graph_build_passes;

  VariantMeasurement() = default;
  VariantMeasurement(VariantMeasurement&&) = default;
  VariantMeasurement& operator=(VariantMeasurement&&) = default;
  VariantMeasurement(const VariantMeasurement&) = delete;
  VariantMeasurement& operator=(const VariantMeasurement&) = delete;
};

// Told when each pass begins, so a live view can show progress. Called on
// the measuring (main) thread.
class MeasurementObserver {
 public:
  virtual ~MeasurementObserver() = default;
  virtual void on_pass_started(const std::string& variant_name,
                               const std::string& pass_description) = 0;
};

// How the timeline pass groups work into timed units. `automatic` times
// stages only when the job names stage cuts and the graph is large enough
// for per-node instrumentation to dominate (see stage_plan.h).
struct TimelineOptions {
  GranularityChoice granularity_choice = GranularityChoice::automatic;
};

// Seven fresh runs per variant: enough for a median that resists one slow
// run, few enough that a millisecond job measures in well under a second.
inline constexpr int JOB_WALL_SAMPLE_COUNT = 7;

// One timeline pass's graph and timeline, kept together. Move-only for the
// same reason as VariantMeasurement: gpu_timeline points into
// timeline_graph.nodes.
struct TimelineCapture {
  CapturedGraph timeline_graph;
  GpuTimeline gpu_timeline;

  TimelineCapture() = default;
  TimelineCapture(TimelineCapture&&) = default;
  TimelineCapture& operator=(TimelineCapture&&) = default;
  TimelineCapture(const TimelineCapture&) = delete;
  TimelineCapture& operator=(const TimelineCapture&) = delete;
};

// Builds and evaluates one fresh graph and returns its outputs, evaluated.
// Its cost is not job cost: the first eval pays Metal kernel compilation
// (and, for a compiled variant, tracing and fused-kernel JIT).
std::vector<mlx::core::array> run_warm_up(const GraphFactory& graph_factory);

// One uninstrumented job-wall sample on a fresh graph, in microseconds.
double sample_job_wall_once(const GraphFactory& graph_factory);

// Median, range and the samples themselves (in measured order) into
// `job_totals`. Throws std::invalid_argument on no samples.
void record_job_wall_samples(const std::vector<double>& job_wall_samples,
                             JobTotals& job_totals);

// One timeline pass on a fresh graph, at the granularity the options and
// the job's stage cuts choose.
TimelineCapture capture_timeline(
    const GraphFactory& graph_factory,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream,
    const TimelineOptions& timeline_options);

// The serial pass on its own fresh graph: fills serial_graph,
// serial_telemetry and the sum of stages.
void run_serial_pass(const GraphFactory& graph_factory,
                     const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream,
                     VariantMeasurement& variant_measurement);

VariantMeasurement measure_variant(
    const std::string& variant_name,
    const GraphFactory& graph_factory,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream,
    MeasurementObserver* measurement_observer = nullptr,
    const TimelineOptions& timeline_options = {});

} // namespace mlx_perf
