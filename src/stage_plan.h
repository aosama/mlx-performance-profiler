#pragma once

// Stage plan: one concern only — decide which captured node belongs to
// which stage, and whether the timeline pass should time nodes or stages.
// Pure functions over topological positions, free of MLX, so the grouping
// rules are unit-tested without a GPU.
//
// Why stages exist: the timeline pass dispatches one async_eval per timed
// unit, and every async_eval commits at least one Metal command buffer. At
// ~1,000 nodes per step (a real decoder) one buffer per node costs several
// times the step itself, so per-node waiting slices mostly measure the
// instrumentation's own queue. Timing a stage (one decoder layer) costs one
// async_eval per stage instead — close to how a plain eval batches work.

#include <string>
#include <string_view>
#include <vector>

namespace mlx_perf {

enum class TimelineGranularity { per_node, per_stage };

// What the caller asks for; `automatic` keeps per-node for small graphs,
// where per-node detail is affordable and most informative.
enum class GranularityChoice { automatic, per_node, per_stage };

// Between the two measured regimes, with margin both ways: the demo jobs
// (tens of nodes) are where per-node detail is the point, while Qwen3.5's
// ~1,100-node decode step costs ~7× its wall when timed per node.
inline constexpr int AUTOMATIC_STAGE_MINIMUM_PRIMITIVE_COUNT = 256;

// `automatic` picks stages only when the job named stage cuts AND the graph
// is large enough for per-node instrumentation to dominate.
TimelineGranularity choose_granularity(
    GranularityChoice granularity_choice, int stage_cut_count, int primitive_count);

std::string_view granularity_name(TimelineGranularity granularity);

// One job-supplied cut, as topological positions of its boundary nodes.
struct StageCut {
  std::string stage_name;
  std::vector<int> boundary_positions;
};

struct StagePlan {
  // Stage names in dispatch order; empty stages are dropped.
  std::vector<std::string> stage_names;
  // Stage index (into stage_names) for every topological position.
  std::vector<int> position_to_stage;
  // Per stage, the members no other member of the same stage consumes —
  // what one async_eval must target to evaluate exactly that stage.
  std::vector<std::vector<int>> stage_to_sink_positions;
};

// Assigns each node to the FIRST cut whose boundary depends on it (Spark's
// rule: a stage owns everything up to its boundary not already computed).
// Every stage's inputs therefore come from itself or earlier stages, so
// dispatching stages in order evaluates each node in its own stage. Nodes
// no cut reaches form a final stage named `final_stage_name`.
// Throws std::invalid_argument on a boundary position out of range or an
// input that does not precede its consumer (not a topological order).
StagePlan plan_stages(
    const std::vector<std::vector<int>>& position_to_input_positions,
    const std::vector<StageCut>& stage_cuts,
    const std::string& final_stage_name);

} // namespace mlx_perf
