#pragma once

// Node work estimate: one concern only — work out, from an MLX operation's
// inputs, outputs and primitive, how many bytes it moves and how much
// arithmetic it does. Runs at capture time, because evaluation detaches a
// node from its inputs and primitive; the one fact only evaluation provides
// (which experts a gathered multiply used) is filled in afterwards.

#include "graph_model.h"
#include "node_work.h"

#include "mlx/array.h"

namespace mlx_perf {

// `operation` must not be evaluated yet. A leaf (no primitive) does no work.
NodeWork estimate_node_work(const mlx::core::array& operation);

// Reads each gathered multiply's expert indices, now evaluated, and narrows
// its bytes_read to the experts it used. Call after a pass has evaluated
// `captured_graph`. Indices the graph does not hold, or that never
// materialized, leave the node counting every expert.
void resolve_gathered_experts(CapturedGraph& captured_graph);

} // namespace mlx_perf
