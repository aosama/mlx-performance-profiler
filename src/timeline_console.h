#pragma once

// Console rendering: one concern only — print the Phase 1 job summary,
// per-stream table and per-node Spark breakdown to stdout, each number next
// to an honest label of what it measures. The HTML version is Phase 3.

#include "gpu_timeline.h"

namespace mlx_plan {

// The job-level totals measured outside gpu_timeline. Together with
// GpuTimeline's own timeline wall they come from three different passes on
// identical, freshly built graphs; they answer different questions and are
// printed side by side, never merged into one "true" duration.
struct JobTotals {
  double job_wall_microseconds = 0.0;         // uninstrumented, warm eval
  double sum_of_stages_microseconds = 0.0;    // serial per-node pass
};

void print_job_summary(const JobTotals& job_totals, const GpuTimeline& gpu_timeline);
void print_stream_table(const GpuTimeline& gpu_timeline);
void print_node_table(const GpuTimeline& gpu_timeline);

} // namespace mlx_plan
