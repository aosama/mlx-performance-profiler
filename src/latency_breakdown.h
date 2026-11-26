#pragma once

// Latency breakdown: one concern only — the Spark StagePage arithmetic that
// splits one node's wall time into waiting slices plus a compute residual.
// Pure functions over timestamps, deliberately free of MLX so the algorithm
// can be unit-tested without a GPU.

#include <string_view>

namespace mlx_perf {

// The four Spark-style buckets. Each maps to a StagePage slice:
//   wait_on_cpu        ~ Scheduler Delay: host time spent issuing the node
//                      (graph walk, kernel encode, command-buffer commit).
//   wait_on_gpu        ~ queueing: issued, but its stream was still busy with
//                      earlier work (or same-stream inputs were not done).
//   scheduler_delay    ~ Scheduler Delay: committed and unblocked, but the GPU
//                      had not begun the command buffer yet (driver/GPU
//                      submission). Only measurable with a GPU window.
//   wait_on_dependency ~ Shuffle Read fetchWaitTime: stream free, but an
//                      input produced on ANOTHER stream had not arrived.
//   compute            ~ Executor Computing Time: the residual.
enum class NodeClass {
  compute,
  wait_on_cpu,
  wait_on_gpu,
  wait_on_dependency,
  scheduler_delay,
};

std::string_view node_class_name(NodeClass node_class);

// Host-clock instants for one node, in microseconds since the pass started.
// "ready" values are 0 when the node has no input of that kind.
struct NodeTimestamps {
  double issue_started_microseconds = 0.0;
  double issue_finished_microseconds = 0.0;
  // Completion of the previous node on the same stream, or of a same-stream
  // input, whichever is later: the earliest the stream could run this node.
  double stream_ready_microseconds = 0.0;
  // Completion of the latest input produced on a different stream.
  double cross_stream_inputs_ready_microseconds = 0.0;
  // When the host observed the node complete (polling).
  double completed_microseconds = 0.0;
  // The GPU's own execution window for the node's command buffer, when the
  // Metal probe measured one. Absent for CPU-stream nodes.
  bool has_gpu_window = false;
  double gpu_started_microseconds = 0.0;
  double gpu_finished_microseconds = 0.0;
};

struct LatencyBreakdown {
  double wait_on_cpu_microseconds = 0.0;
  double wait_on_gpu_microseconds = 0.0;
  double wait_on_dependency_microseconds = 0.0;
  double scheduler_delay_microseconds = 0.0;
  double compute_microseconds = 0.0;
  // GPU end → host-observed completion: Spark's "Getting Result Time". It is
  // observation cost, so it is reported but never decides the class.
  double completion_latency_microseconds = 0.0;
  // True when compute is GPU-measured (GPUEndTime − GPUStartTime, minus any
  // in-buffer dependency wait) rather than the host-window residual.
  bool is_compute_measured = false;
  // completed − issue_started; equals the sum of the six slices.
  double wall_microseconds = 0.0;
  NodeClass classification = NodeClass::compute;
};

// Walks the node's window [issue_started, completed] forward through the
// blocking reasons in causal order (issue → stream free → remote data →
// GPU scheduling → execution → host notices), so slices never overlap and
// always sum to the wall — Spark's invariant. With a GPU window, compute is
// measured; without one it is the residual `compute = wall − waits`.
LatencyBreakdown decompose_node_latency(const NodeTimestamps& timestamps);

// The dominant slice names the node; ties resolve toward compute so a node is
// only called "waiting" when waiting strictly outweighs working.
NodeClass classify_breakdown(const LatencyBreakdown& breakdown);

} // namespace mlx_perf
