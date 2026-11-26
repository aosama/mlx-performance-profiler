#include "timed_units.h"

#include <algorithm>
#include <map>

namespace mlx_perf {
namespace {

std::string node_caption(const NodeTimeline& node_timeline) {
  return node_timeline.node->label + " " + node_timeline.node->op_name +
      " (stream " + std::to_string(node_timeline.node->stream_index) + ")";
}

std::string stage_caption(const StageTimeline& stage_timeline) {
  return stage_timeline.stage_name + " (stream " +
      std::to_string(stage_timeline.stream_index) + ", " +
      std::to_string(stage_timeline.node_positions.size()) + " nodes)";
}

std::vector<TimedUnit> collect_node_units(const GpuTimeline& gpu_timeline) {
  std::vector<TimedUnit> timed_units;
  for (const NodeTimeline& node_timeline : gpu_timeline.nodes) {
    timed_units.push_back({node_timeline.node->stream_index, 1,
                           node_caption(node_timeline), &node_timeline.timestamps,
                           &node_timeline.breakdown,
                           node_timeline.cross_stream_producer_positions});
  }
  return timed_units;
}

std::vector<TimedUnit> collect_stage_units(const GpuTimeline& gpu_timeline) {
  std::vector<TimedUnit> timed_units;
  for (const StageTimeline& stage_timeline : gpu_timeline.stages) {
    timed_units.push_back({stage_timeline.stream_index,
                           static_cast<int>(stage_timeline.node_positions.size()),
                           stage_caption(stage_timeline), &stage_timeline.timestamps,
                           &stage_timeline.breakdown,
                           stage_timeline.cross_stream_producer_stage_positions});
  }
  return timed_units;
}

void accumulate_into_stream(
    const TimedUnit& timed_unit, StreamTimeline& stream_timeline) {
  if (stream_timeline.node_count == 0) {
    stream_timeline.first_issue_microseconds =
        timed_unit.timestamps->issue_started_microseconds;
  }
  stream_timeline.node_count += timed_unit.node_count;
  stream_timeline.first_issue_microseconds =
      std::min(stream_timeline.first_issue_microseconds,
               timed_unit.timestamps->issue_started_microseconds);
  stream_timeline.last_completion_microseconds =
      std::max(stream_timeline.last_completion_microseconds,
               timed_unit.timestamps->completed_microseconds);
  stream_timeline.busy_microseconds += timed_unit.breakdown->compute_microseconds;
  stream_timeline.dependency_wait_microseconds +=
      timed_unit.breakdown->wait_on_dependency_microseconds;
}

} // namespace

std::vector<TimedUnit> collect_timed_units(const GpuTimeline& gpu_timeline) {
  return gpu_timeline.granularity == TimelineGranularity::per_stage
      ? collect_stage_units(gpu_timeline)
      : collect_node_units(gpu_timeline);
}

std::vector<StreamTimeline> summarize_streams(
    const std::vector<TimedUnit>& timed_units,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream) {
  std::map<int, StreamTimeline> stream_index_to_timeline;
  for (const TimedUnit& timed_unit : timed_units) {
    StreamTimeline& stream_timeline = stream_index_to_timeline[timed_unit.stream_index];
    stream_timeline.stream_index = timed_unit.stream_index;
    stream_timeline.is_gpu =
        stream_index_to_stream.at(timed_unit.stream_index).device ==
        mlx::core::Device::gpu;
    accumulate_into_stream(timed_unit, stream_timeline);
  }
  std::vector<StreamTimeline> stream_timelines;
  for (auto& [stream_index, stream_timeline] : stream_index_to_timeline) {
    stream_timeline.wall_microseconds = stream_timeline.last_completion_microseconds -
        stream_timeline.first_issue_microseconds;
    stream_timelines.push_back(stream_timeline);
  }
  return stream_timelines;
}

LatencyBreakdown aggregate_breakdown(const std::vector<TimedUnit>& timed_units) {
  LatencyBreakdown aggregate;
  for (const TimedUnit& timed_unit : timed_units) {
    const LatencyBreakdown& breakdown = *timed_unit.breakdown;
    aggregate.wait_on_cpu_microseconds += breakdown.wait_on_cpu_microseconds;
    aggregate.wait_on_gpu_microseconds += breakdown.wait_on_gpu_microseconds;
    aggregate.wait_on_dependency_microseconds +=
        breakdown.wait_on_dependency_microseconds;
    aggregate.scheduler_delay_microseconds += breakdown.scheduler_delay_microseconds;
    aggregate.compute_microseconds += breakdown.compute_microseconds;
    aggregate.completion_latency_microseconds +=
        breakdown.completion_latency_microseconds;
    aggregate.wall_microseconds += breakdown.wall_microseconds;
  }
  return aggregate;
}

const char* timed_unit_noun(const GpuTimeline& gpu_timeline) {
  return gpu_timeline.granularity == TimelineGranularity::per_stage ? "stage" : "node";
}

} // namespace mlx_perf
