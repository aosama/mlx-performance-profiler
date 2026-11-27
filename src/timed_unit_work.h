#pragma once

// Timed unit work: one concern only — the work each timed unit did, and its
// judgement. A node's work is its capture-time estimate, unless the serial
// pass shows it launched no kernel (a reshape or a broadcast only describes
// a view, and moves nothing). A stage task's work is the sum of its members'.

#include "gpu_timeline.h"
#include "machine_ceilings.h"
#include "node_work.h"
#include "telemetry.h"
#include "work_limits.h"

#include <cstddef>

namespace mlx_perf {

NodeWork work_of_node(const GpuTimeline& gpu_timeline, std::size_t position,
                      const RunTelemetry& serial_telemetry);

NodeWork work_of_stage(const GpuTimeline& gpu_timeline, const StageTimeline& stage_task,
                       const RunTelemetry& serial_telemetry);

// A node judged against its own GPU window. In a stage-timed run a node has
// no window of its own and is not judged; its stage is.
WorkJudgement judge_node(const GpuTimeline& gpu_timeline, std::size_t position,
                         const RunTelemetry& serial_telemetry, const MachineCeilings& machine_ceilings);

WorkJudgement judge_stage(const GpuTimeline& gpu_timeline, const StageTimeline& stage_task,
                          const RunTelemetry& serial_telemetry, const MachineCeilings& machine_ceilings);

} // namespace mlx_perf
