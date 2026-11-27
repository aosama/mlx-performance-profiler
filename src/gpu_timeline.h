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
//  - Per-node async_eval commits one GPU submission per node (a Metal command
//    buffer, a CUDA graph launch), which a plain eval batches. The timeline wall therefore exceeds the
//    uninstrumented job wall; report both, never one as the whole truth. On
//    ~1,000-node graphs that cost dominates (Qwen3.5 decode: 23 ms vs a
//    3 ms step), so stage_timeline times job-supplied stages instead; this
//    struct carries either granularity.
//  - GPU execution IS measured (Phase 2): metal_command_buffer_probe reads
//    each node's command-buffer GPUStartTime/GPUEndTime on the host clock,
//    and producer readiness uses those GPU end times. Granularity is one
//    command buffer = one node here; CPU-stream nodes fall back to the host
//    residual and are marked unmeasured.
//  - Streams are modeled as in-order queues (MLX encodes one queue per
//    stream), so a node cannot start before its stream's previous node ends.

#include "graph_model.h"
#include "latency_breakdown.h"
#include "stage_plan.h"

#include "mlx/stream.h"

#include <unordered_map>
#include <vector>

namespace mlx_perf {

struct NodeTimeline {
  const GraphNode* node = nullptr;
  NodeTimestamps timestamps;   // µs since the pass started
  LatencyBreakdown breakdown;  // Spark slices of [issue_started, completed]
  bool has_cross_stream_input = false;  // a join in the Spark-shuffle sense
  // Positions (in GpuTimeline::nodes) of producers on other streams — the
  // cross-stream edges a Gantt draws as dependency arrows.
  std::vector<int> cross_stream_producer_positions;
  // Every producer this node reads (both streams), and the node before it on
  // its own stream (NO_POSITION when first): the structure a data consumer
  // needs to compute gaps between nodes without re-deriving the graph.
  std::vector<int> input_positions;
  int previous_on_stream_position = -1;  // NO_POSITION (timeline_dependencies.h)
  // Per-stage pass only: the position (in GpuTimeline::stages) of the stage
  // task this node ran in. Its own timestamps/breakdown stay zero there —
  // the node was not timed individually, and must not pretend it was.
  int stage_position = -1;
};

// One stage on one stream: Spark's task of a stage (a stage runs as one task
// per executor; here, per stream it touches). Its window covers every
// member on that stream, dispatched by ONE async_eval.
struct StageTimeline {
  int stage_index = 0;          // into the stage plan's dispatch order
  std::string stage_name;       // job-supplied, e.g. "layer 3"
  int stream_index = 0;
  std::vector<int> node_positions;  // members, positions in GpuTimeline::nodes
  NodeTimestamps timestamps;
  LatencyBreakdown breakdown;
  // Stage tasks on other streams, in EARLIER stages, this task reads from.
  // Cross-stream reads inside one stage are fences within the window and
  // count as its compute: per-stage timing cannot separate them.
  std::vector<int> cross_stream_producer_stage_positions;
  bool has_cross_stream_input = false;
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
  TimelineGranularity granularity = TimelineGranularity::per_node;
  // One entry per primitive, in topological order. Per-node: each is a timed
  // unit. Per-stage: identity and stage membership only.
  std::vector<NodeTimeline> nodes;
  // Per-stage pass only: the timed units, in dispatch order.
  std::vector<StageTimeline> stages;
  std::vector<StreamTimeline> streams;
  // First issue → last completion of this instrumented pass.
  double timeline_wall_microseconds = 0.0;
  // Sum of per-node wait_on_dependency: stream-time spent blocked on another
  // stream's data, Spark's summed fetchWaitTime. Concurrent stalls on
  // different streams add, so this can exceed the wall.
  double cross_stream_stall_microseconds = 0.0;
  // Sum of GPU-measured compute, and how many nodes had a GPU window (in the
  // per-stage pass: how many nodes sat in a stage task with one).
  double measured_gpu_compute_microseconds = 0.0;
  int gpu_measured_node_count = 0;
  // MLX command buffers the probe held and saw complete in this pass. Exact
  // per node (one buffer per async_eval); a lower bound per stage, where MLX
  // may commit mid-stage buffers the probe never holds.
  int observed_command_buffer_count = 0;
  bool is_command_buffer_count_lower_bound = false;
};

// The returned NodeTimeline::node pointers point into `captured_graph`, so it
// must outlive the GpuTimeline.
// `captured_graph` must be captured from arrays that are NOT yet evaluated:
// MLX detaches inputs once a node is evaluated, so dependencies are
// snapshotted here before the first dispatch. Evaluates the graph.
GpuTimeline capture_gpu_timeline(
    const CapturedGraph& captured_graph,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream);

} // namespace mlx_perf
