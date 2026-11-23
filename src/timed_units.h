#pragma once

// Timed units: one concern only — present what a timeline pass actually
// timed (one unit per node, or one per stage task) through a single view,
// so summaries, stream tables and class counts never branch on granularity
// and can never sum per-node zeros of a per-stage pass by mistake.

#include "gpu_timeline.h"

#include "mlx/stream.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_plan {

struct TimedUnit {
  int stream_index = 0;
  int node_count = 1;  // nodes the unit covers (1 per node; members per stage task)
  std::string caption;  // "B MatMul (stream 1)" or "layer 3 (stream 1, 47 nodes)"
  const NodeTimestamps* timestamps = nullptr;
  const LatencyBreakdown* breakdown = nullptr;
  // Positions (in the returned vector) of cross-stream producer units.
  std::vector<int> cross_stream_producer_units;
};

// Points into `gpu_timeline`, which must outlive the returned units.
std::vector<TimedUnit> collect_timed_units(const GpuTimeline& gpu_timeline);

// Per-stream wall/busy/stall totals over the timed units; std::map order
// (by stream index) for stable reporting.
std::vector<StreamTimeline> summarize_streams(
    const std::vector<TimedUnit>& timed_units,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream);

// Sums every unit's slices: the job-level "what was the work doing".
LatencyBreakdown aggregate_breakdown(const std::vector<TimedUnit>& timed_units);

// "node" or "stage" — the noun the reports use for one timed unit.
const char* timed_unit_noun(const GpuTimeline& gpu_timeline);

} // namespace mlx_plan
