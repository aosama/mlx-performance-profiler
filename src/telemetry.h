#pragma once

// Telemetry: the "what actually ran" layer Spark's stage table is made of.
// Two passes, because per-node attribution and true pipeline wall-clock are
// different questions — Spark reports both the same way (stage durations vs
// job duration).

#include "graph_model.h"

#include "mlx/stream.h"

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace mlx_plan {

struct NodeTelemetry {
  const GraphNode* node = nullptr;
  double duration_microseconds = 0.0;
  // Bytes of memory newly active after this node's eval; approximate because
  // the capture keeps every intermediate alive and Metal may reuse buffers.
  size_t memory_added_bytes = 0;
  size_t output_bytes = 0;
};

struct RunTelemetry {
  // One entry per primitive node, in topological (execution) order.
  std::vector<NodeTelemetry> attributed_nodes;
  // Sum of per-node durations from the serialized measurement pass.
  double attributed_total_microseconds = 0.0;
  // Wall-clock of a separate, uninstrumented single-eval pass on a rebuilt
  // graph — the honest end-to-end number.
  double pipeline_total_microseconds = 0.0;
  size_t run_peak_memory_bytes = 0;
  int node_count = 0;
};

// `stream_index_to_stream` supplies the MLX Stream handles the graph was built
// with, so each node can be synchronized on the stream it actually runs on.
RunTelemetry run_attributed_pass(
    const CapturedGraph& captured_graph,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream);

} // namespace mlx_plan
