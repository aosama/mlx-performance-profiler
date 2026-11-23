#pragma once

// Job-level totals measured outside gpu_timeline. Together with GpuTimeline's
// own timeline wall they come from three different passes on identical,
// freshly built graphs; they answer different questions and are shown side
// by side, never merged into one "true" duration.

namespace mlx_plan {

struct JobTotals {
  double job_wall_microseconds = 0.0;       // uninstrumented, warm eval
  double sum_of_stages_microseconds = 0.0;  // serial per-node pass
};

} // namespace mlx_plan
