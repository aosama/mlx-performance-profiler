#pragma once

// Console rendering: one concern only — print the Phase 1 job summary,
// per-stream table and per-node (or per-stage) Spark breakdown to stdout, each number next
// to an honest label of what it measures. The HTML version is Phase 3.

#include "gpu_timeline.h"
#include "job_totals.h"

namespace mlx_plan {

void print_job_summary(const JobTotals& job_totals, const GpuTimeline& gpu_timeline);
void print_stream_table(const GpuTimeline& gpu_timeline);
void print_node_table(const GpuTimeline& gpu_timeline);
// One row per stage task; prints nothing for a per-node timeline.
void print_stage_table(const GpuTimeline& gpu_timeline);

} // namespace mlx_plan
