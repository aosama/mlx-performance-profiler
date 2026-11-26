#pragma once

// Report inputs: the measured results every report section reads. A bundle
// of references, not owned data — built by the orchestrator right before
// rendering, while the graphs and timelines it points at are alive.

#include "gpu_timeline.h"
#include "graph_model.h"
#include "job_totals.h"
#include "telemetry.h"

#include <string>

namespace mlx_perf {

struct ReportInputs {
  std::string report_title;
  // The graph the timeline pass measured; GpuTimeline nodes point into it.
  const CapturedGraph& timeline_graph;
  const GpuTimeline& gpu_timeline;
  // The serial pass, on an identical graph: per-node isolated durations and
  // peak memory. Joined to the timeline by topological position.
  const RunTelemetry& serial_telemetry;
  JobTotals job_totals;
};

} // namespace mlx_perf
