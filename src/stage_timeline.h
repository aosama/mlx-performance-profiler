#pragma once

// Stage timeline: one concern only — the overlap-preserving timeline pass at
// STAGE granularity. Same measurement model as gpu_timeline (async_eval,
// host-observed completion, Metal command-buffer windows, the Spark
// decomposition), but one async_eval per stage instead of per node, so the
// instrumentation stops dominating graphs of ~1,000 nodes.
//
// What a stage task's numbers mean (MLX 0.32.4, transforms.cpp eval_impl):
//  - One async_eval targets the stage's sink nodes; MLX encodes the whole
//    stage, attaching one Event per stream, so every member on a stream
//    becomes available together. Completion is per stage task, never per node.
//  - The GPU window runs from the GPU start of the stream's first command
//    buffer of the stage to the END of its last one, bounded by an empty
//    marker buffer committed behind the stage on the same queue (MLX commits
//    mid-stage when a buffer exceeds its op/MB budget, and those middle
//    buffers are invisible to the probe). The end is the smaller of the
//    marker's GPU start and host-observed completion: both can only trail
//    the true end, never lead it.
//  - "compute" therefore includes any gap where the GPU waited for the host
//    to commit the stage's next buffer: per-stage timing cannot separate
//    GPU work from mid-stage host encoding. The host-issue slice still shows
//    how long encoding took.
//  - Cross-stream reads between stages are stalls (counted until all such
//    inputs are ready, an upper bound when a stage starts independent work
//    first); cross-stream reads inside a stage are fences within its window.

#include "gpu_timeline.h"
#include "graph_factory.h"
#include "stage_plan.h"

#include "mlx/stream.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_plan {

// Translates job-supplied boundary arrays into cuts over `captured_graph`'s
// topological positions. Boundary arrays not in the graph (already
// materialized, or not on the path to any output) are skipped.
std::vector<StageCut> resolve_stage_cuts(
    const CapturedGraph& captured_graph,
    const std::vector<StageBoundary>& stage_boundaries);

// Same contract as capture_gpu_timeline (unevaluated graph, evaluates it,
// node pointers into `captured_graph`, throws std::out_of_range before any
// dispatch for an unknown stream). Nodes no cut reaches form a final stage
// named `final_stage_name`; with no cuts the whole job is one stage.
GpuTimeline capture_stage_timeline(
    const CapturedGraph& captured_graph,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream,
    const std::vector<StageCut>& stage_cuts,
    const std::string& final_stage_name);

} // namespace mlx_plan
