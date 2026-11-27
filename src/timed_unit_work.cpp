#include "timed_unit_work.h"

namespace mlx_perf {

NodeWork work_of_node(const GpuTimeline& gpu_timeline, std::size_t position,
                      const RunTelemetry& serial_telemetry) {
  const GraphNode& graph_node = *gpu_timeline.nodes[position].node;
  const NodeTelemetry* serial_node =
      find_attributed_node(serial_telemetry, position, graph_node.op_name);
  // An unmatched serial node says nothing about launches: keep the estimate
  // rather than drop work that may have run.
  if (serial_node != nullptr && serial_node->kernel_dispatches.empty()) {
    return NodeWork{};
  }
  return graph_node.work;
}

NodeWork work_of_stage(const GpuTimeline& gpu_timeline, const StageTimeline& stage_task,
                       const RunTelemetry& serial_telemetry) {
  NodeWork stage_work;
  for (const int member_position : stage_task.node_positions) {
    add_node_work(stage_work, work_of_node(gpu_timeline, static_cast<std::size_t>(member_position),
                                           serial_telemetry));
  }
  return stage_work;
}

WorkJudgement judge_node(const GpuTimeline& gpu_timeline, std::size_t position,
                         const RunTelemetry& serial_telemetry, const MachineCeilings& machine_ceilings) {
  if (gpu_timeline.granularity != TimelineGranularity::per_node) {
    return WorkJudgement{};
  }
  const LatencyBreakdown& breakdown = gpu_timeline.nodes[position].breakdown;
  return judge_work(work_of_node(gpu_timeline, position, serial_telemetry), machine_ceilings,
                    breakdown.compute_microseconds, breakdown.is_compute_measured);
}

WorkJudgement judge_stage(const GpuTimeline& gpu_timeline, const StageTimeline& stage_task,
                          const RunTelemetry& serial_telemetry, const MachineCeilings& machine_ceilings) {
  return judge_work(work_of_stage(gpu_timeline, stage_task, serial_telemetry), machine_ceilings,
                    stage_task.breakdown.compute_microseconds, stage_task.breakdown.is_compute_measured);
}

} // namespace mlx_perf
