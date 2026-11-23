#pragma once

// Measurement session: one concern only — run the four passes (warm-up,
// job wall, timeline, serial) for one job variant and keep everything the
// reports need alive together.

#include "gpu_timeline.h"
#include "graph_factory.h"
#include "graph_model.h"
#include "job_totals.h"
#include "telemetry.h"

#include "mlx/stream.h"

#include <string>
#include <unordered_map>

namespace mlx_plan {

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

  VariantMeasurement() = default;
  VariantMeasurement(VariantMeasurement&&) = default;
  VariantMeasurement& operator=(VariantMeasurement&&) = default;
  VariantMeasurement(const VariantMeasurement&) = delete;
  VariantMeasurement& operator=(const VariantMeasurement&) = delete;
};

VariantMeasurement measure_variant(
    const std::string& variant_name,
    const GraphFactory& graph_factory,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream);

} // namespace mlx_plan
