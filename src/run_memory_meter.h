#pragma once

// Run memory meter: one concern only — what one measured run did to memory,
// read just before it and just after it, never during: MLX's own peak and
// what it left behind, the system's wired total at its end, and whether the
// system crossed the wire limit, compressed or swapped meanwhile (#46).
//
// Why only before and after: these runs are the judged rounds, and any
// reading inside one would add its cost to the time being judged, more for
// the procedure that evaluates more often. So two figures are not what the
// issue first asked for: MLX does not track a peak of active plus cache, so
// the run's end holding stands in; and the system's wired peak needs
// sampling, so the wired total at the run's end stands in (the observed
// run's wired memory lane has both true peaks).

#include <cstdint>
#include <vector>

namespace mlx_perf {

struct RunMemoryFigures {
  // MLX's peak active memory during the run (MLX's own peak counter, reset
  // as the run starts), and that peak less the active memory at the start:
  // the run's own working set. The second is the one to compare, because
  // MLX's counters are process-wide: memory another procedure keeps alive
  // between its runs is in this run's raw peak too.
  std::uint64_t peak_mlx_active_bytes = 0;
  std::uint64_t peak_mlx_active_added_bytes = 0;
  // MLX's active memory plus cache at the end, and less at the start: what
  // the run left behind (negative when it freed more than it kept).
  std::uint64_t mlx_held_at_end_bytes = 0;
  std::int64_t mlx_held_change_bytes = 0;
  std::uint64_t system_wired_at_end_bytes = 0;
  std::uint64_t wire_limit_bytes = 0;  // 0 when unknown
  // System-wide during the run, from the kernel's running counters.
  std::uint64_t wire_requests_over_limit_count = 0;
  std::uint64_t compressed_bytes = 0;
  std::int64_t swap_used_change_bytes = 0;
  bool are_pressure_counters_readable = false;
};

// The run went over the wire limit: the kernel refused wire requests over it,
// or the wired total was over it at the run's end.
bool has_crossed_wire_limit(const RunMemoryFigures& run_memory_figures);

// The system compressed memory during the run (any program's memory).
bool has_compressed_memory(const RunMemoryFigures& run_memory_figures);

// Over several runs' figures: how many crossed the limit or compressed.
int count_runs_crossing_wire_limit(const std::vector<RunMemoryFigures>& run_memory_figures);
int count_runs_with_compression(const std::vector<RunMemoryFigures>& run_memory_figures);

// One figure of each run, in bytes, as doubles for the paired comparison.
std::vector<double> collect_run_bytes(const std::vector<RunMemoryFigures>& run_memory_figures,
                                      std::uint64_t RunMemoryFigures::*byte_field);
std::vector<double> collect_run_bytes(const std::vector<RunMemoryFigures>& run_memory_figures,
                                      std::int64_t RunMemoryFigures::*byte_field);

class RunMemoryMeter {
 public:
  // Resets MLX's peak counter and reads the starting figures. Call right
  // before the timed run, after its setup has finished on the GPU.
  void start();
  // Reads the ending figures. Call right after the timed run's GPU work has
  // completed.
  RunMemoryFigures finish() const;

 private:
  std::uint64_t mlx_active_at_start_bytes_ = 0;
  std::uint64_t mlx_held_at_start_bytes_ = 0;
  std::uint64_t wire_requests_over_limit_at_start_ = 0;
  std::uint64_t compressor_input_bytes_at_start_ = 0;
  std::uint64_t swap_used_bytes_at_start_ = 0;
  bool are_pressure_counters_readable_at_start_ = false;
};

} // namespace mlx_perf
