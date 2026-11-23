#pragma once

// Job-level totals measured outside gpu_timeline. Together with GpuTimeline's
// own timeline wall they come from three different passes on identical,
// freshly built graphs; they answer different questions and are shown side
// by side, never merged into one "true" duration.

namespace mlx_plan {

struct JobTotals {
  // Uninstrumented, warm eval: the MEDIAN of several fresh runs. One run of a
  // millisecond job swings by tens of percent (GPU clocks, scheduling), too
  // much to compare variants on; the range is kept so the spread stays visible.
  double job_wall_microseconds = 0.0;
  double job_wall_min_microseconds = 0.0;
  double job_wall_max_microseconds = 0.0;
  int job_wall_sample_count = 0;
  double sum_of_stages_microseconds = 0.0;  // serial per-node pass
};

// True when `other_wall_microseconds` falls inside the sampled job-wall
// range: the difference is then run-to-run noise, not a measurable effect.
inline bool is_within_job_wall_range(
    const JobTotals& job_totals, double other_wall_microseconds) {
  return job_totals.job_wall_sample_count > 0 &&
      other_wall_microseconds >= job_totals.job_wall_min_microseconds &&
      other_wall_microseconds <= job_totals.job_wall_max_microseconds;
}

} // namespace mlx_plan
