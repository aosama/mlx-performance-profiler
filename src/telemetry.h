#pragma once

// Telemetry: the "what actually ran" layer Spark's stage table is made of.
// Owns the two serial-or-whole passes — per-node attribution (sum-of-stages)
// and the uninstrumented pipeline wall-clock — because they are different
// questions, as Spark separates stage durations from job duration. The
// overlap-preserving per-node pass lives in gpu_timeline.

#include "dispatch_occupancy.h"
#include "gpu_device_facts.h"
#include "graph_model.h"
#include "wired_memory_sample.h"

#include "mlx/stream.h"

#include <cstddef>
#include <unordered_map>
#include <string>
#include <vector>

namespace mlx_perf {

struct NodeTelemetry {
  const GraphNode* node = nullptr;
  double duration_microseconds = 0.0;
  // Bytes of memory newly active after this node's eval; approximate because
  // the capture keeps every intermediate alive and Metal may reuse buffers.
  size_t memory_added_bytes = 0;
  size_t output_bytes = 0;
  // Every kernel launch this node's evaluation encoded, in order. Recorded
  // here because the serial pass evaluates each node alone, so the
  // attribution is exact in per-node and per-stage runs alike.
  std::vector<KernelDispatch> kernel_dispatches;
};

struct RunTelemetry {
  // One entry per primitive node, in topological (execution) order.
  std::vector<NodeTelemetry> attributed_nodes;
  // Sum of per-node durations from the serialized measurement pass. The
  // uninstrumented job wall lives in JobTotals, not here: it is a different
  // pass and has exactly one home.
  double attributed_total_microseconds = 0.0;
  // Highest per-node peak during the pass, above the memory already active
  // when the pass began.
  size_t run_peak_memory_bytes = 0;
  int node_count = 0;
  // The GPU the pass ran on: its core count turns each recorded kernel
  // launch's threadgroups into the share of the GPU the launch could use.
  GpuDeviceFacts gpu_device_facts;
  // The system's wired memory beside MLX's (#45): one sample as the pass
  // began, then one after each node, so sample k + 1 follows the node at
  // position k. The pass keeps every intermediate alive, so MLX's holding
  // only grows here; the samples show how close the job runs to the wire
  // limit, not what an engine's releases look like.
  std::vector<WiredMemorySample> wired_memory_samples;
};

// `stream_index_to_stream` supplies the MLX Stream handles the graph was built
// with, so each node can be synchronized on the stream it actually runs on.
RunTelemetry run_attributed_pass(
    const CapturedGraph& captured_graph,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream);

// One uninstrumented run of the job, in microseconds.
struct PipelinePassTiming {
  // First issue to every output observed complete: the honest job duration.
  double job_wall_microseconds = 0.0;
  // Host time inside the single async_eval: walking, encoding and committing
  // the whole job, with no per-node instrumentation in the way.
  double host_submission_microseconds = 0.0;
};

// The honest end-to-end number: evaluates `output_nodes` once with a single
// async_eval (MLX batches the whole graph, streams overlap) and observes
// completion by polling like the timeline pass. Run it on a freshly built
// graph, and only after a warm-up run: the first eval in a process pays
// one-time GPU library load and kernel compilation, which is not the
// job's cost.
PipelinePassTiming run_pipeline_pass(std::vector<mlx::core::array> output_nodes);

// The serial pass ran an identical graph, so topological position joins it
// to the timeline pass; the op name guards the join, and a mismatch returns
// nullptr — never a number borrowed from a different node.
const NodeTelemetry* find_attributed_node(
    const RunTelemetry& serial_telemetry,
    std::size_t position,
    const std::string& expected_op_name);

} // namespace mlx_perf
