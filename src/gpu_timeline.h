#pragma once

// GPU timeline: one concern only — a NON-serializing measurement pass that
// records, per node, when the host issued it and when the host observed it
// complete, with every stream still overlapping. The Spark decomposition of
// those instants lives in latency_breakdown; this module only measures.
//
// How (grounded in MLX 0.32.4 transforms.cpp): each primitive is dispatched
// with its own `async_eval`, in topological order. async_eval returns without
// waiting, attaches a fresh per-stream Event to the node, and makes any
// cross-stream input a GPU-side event wait — so the host never blocks and the
// streams run concurrently. The host then polls `array::is_available()`
// (non-blocking: status + Event::is_signaled) to timestamp completions.
//
// Honest limits of these numbers:
//  - Completion is HOST-OBSERVED: it trails true kernel end by the event
//    signal latency plus the polling gap (one node's issue time while still
//    dispatching; a spin-yield afterwards).
//  - Per-node async_eval commits one Metal command buffer per node, which a
//    plain eval batches. The timeline wall therefore exceeds the
//    uninstrumented job wall; report both, never one as the whole truth.
//  - Absolute GPU kernel nanoseconds are not measured here: Event::value()
//    is a monotonic counter, not a clock. Kernel-accurate time is Phase 2
//    (Metal capture). "compute" below is window residual, i.e. approximate.
//  - Streams are modeled as in-order queues (MLX encodes one queue per
//    stream), so a node cannot start before its stream's previous node ends.

#include "graph_model.h"
#include "latency_breakdown.h"

#include "mlx/stream.h"

#include <unordered_map>
#include <vector>

namespace mlx_plan {

struct NodeTimeline {
  const GraphNode* node = nullptr;
  NodeTimestamps timestamps;   // µs since the pass started
  LatencyBreakdown breakdown;  // Spark slices of [issue_started, completed]
  bool has_cross_stream_input = false;  // a join in the Spark-shuffle sense
};

struct StreamTimeline {
  int stream_index = 0;
  bool is_gpu = true;
  int node_count = 0;
  double first_issue_microseconds = 0.0;
  double last_completion_microseconds = 0.0;
  double wall_microseconds = 0.0;  // last completion − first issue
  // Sum of compute residuals. Within one in-order stream they cannot
  // overlap, so wall − busy is time the stream sat idle or blocked.
  double busy_microseconds = 0.0;
  double dependency_wait_microseconds = 0.0;
};

struct GpuTimeline {
  // One entry per primitive, in topological (dispatch) order.
  std::vector<NodeTimeline> nodes;
  std::vector<StreamTimeline> streams;
  // First issue → last completion of this instrumented pass.
  double timeline_wall_microseconds = 0.0;
  // Sum of per-node wait_on_dependency: stream-time spent blocked on another
  // stream's data, Spark's summed fetchWaitTime. Concurrent stalls on
  // different streams add, so this can exceed the wall.
  double cross_stream_stall_microseconds = 0.0;
};

// The returned NodeTimeline::node pointers point into `captured_graph`, so it
// must outlive the GpuTimeline.
// `captured_graph` must be captured from arrays that are NOT yet evaluated:
// MLX detaches inputs once a node is evaluated, so dependencies are
// snapshotted here before the first dispatch. Evaluates the graph.
GpuTimeline capture_gpu_timeline(
    const CapturedGraph& captured_graph,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream);

} // namespace mlx_plan
