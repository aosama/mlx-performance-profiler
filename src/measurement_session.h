#pragma once

// Measurement session: one concern only — run the four passes (warm-up,
// job wall, timeline at the chosen granularity, serial) for one job variant and keep everything the
// reports need alive together.

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

VariantMeasurement measure_variant(
    const std::string& variant_name,
    const GraphFactory& graph_factory,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream,
    MeasurementObserver* measurement_observer = nullptr,
    const TimelineOptions& timeline_options = {});

} // namespace mlx_perf
