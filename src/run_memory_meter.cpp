#include "run_memory_meter.h"

#include "system_memory_pressure.h"
#include "system_wired_memory.h"

#include "mlx/memory.h"

namespace mlx_perf {
namespace {

std::uint64_t read_mlx_held_bytes() {
  return static_cast<std::uint64_t>(mlx::core::get_active_memory()) +
         static_cast<std::uint64_t>(mlx::core::get_cache_memory());
}

// How far a figure rose, never below zero: running counters only grow (one
// that went backwards, a kernel reset, reads as no activity rather than as a
// huge unsigned difference), and a peak is never below its starting level.
std::uint64_t counter_increase(std::uint64_t value_at_start, std::uint64_t value_at_end) {
  return value_at_end > value_at_start ? value_at_end - value_at_start : 0;
}

} // namespace

bool has_crossed_wire_limit(const RunMemoryFigures& run_memory_figures) {
  const bool is_over_limit_at_end = run_memory_figures.wire_limit_bytes > 0 &&
                                    run_memory_figures.system_wired_at_end_bytes > run_memory_figures.wire_limit_bytes;
  return run_memory_figures.wire_requests_over_limit_count > 0 || is_over_limit_at_end;
}

bool has_compressed_memory(const RunMemoryFigures& run_memory_figures) {
  return run_memory_figures.compressed_bytes > 0;
}

int count_runs_crossing_wire_limit(const std::vector<RunMemoryFigures>& run_memory_figures) {
  int crossing_run_count = 0;
  for (const RunMemoryFigures& memory_figures : run_memory_figures) {
    crossing_run_count += has_crossed_wire_limit(memory_figures) ? 1 : 0;
  }
  return crossing_run_count;
}

int count_runs_with_compression(const std::vector<RunMemoryFigures>& run_memory_figures) {
  int compressing_run_count = 0;
  for (const RunMemoryFigures& memory_figures : run_memory_figures) {
    compressing_run_count += has_compressed_memory(memory_figures) ? 1 : 0;
  }
  return compressing_run_count;
}

std::vector<double> collect_run_bytes(const std::vector<RunMemoryFigures>& run_memory_figures,
                                      std::uint64_t RunMemoryFigures::*byte_field) {
  std::vector<double> run_bytes;
  for (const RunMemoryFigures& memory_figures : run_memory_figures) {
    run_bytes.push_back(static_cast<double>(memory_figures.*byte_field));
  }
  return run_bytes;
}

std::vector<double> collect_run_bytes(const std::vector<RunMemoryFigures>& run_memory_figures,
                                      std::int64_t RunMemoryFigures::*byte_field) {
  std::vector<double> run_bytes;
  for (const RunMemoryFigures& memory_figures : run_memory_figures) {
    run_bytes.push_back(static_cast<double>(memory_figures.*byte_field));
  }
  return run_bytes;
}

void RunMemoryMeter::start() {
  const MemoryPressureCounters pressure_counters = read_memory_pressure_counters();
  wire_requests_over_limit_at_start_ = pressure_counters.wire_requests_over_limit_count;
  compressor_input_bytes_at_start_ = pressure_counters.compressor_input_bytes;
  swap_used_bytes_at_start_ = pressure_counters.swap_used_bytes;
  are_pressure_counters_readable_at_start_ = pressure_counters.is_readable;
  mlx_active_at_start_bytes_ = mlx::core::get_active_memory();
  mlx_held_at_start_bytes_ = read_mlx_held_bytes();
  // Last: from here MLX's peak counter covers only the run itself.
  mlx::core::reset_peak_memory();
}

RunMemoryFigures RunMemoryMeter::finish() const {
  RunMemoryFigures run_memory_figures;
  run_memory_figures.peak_mlx_active_bytes = mlx::core::get_peak_memory();
  run_memory_figures.peak_mlx_active_added_bytes =
      counter_increase(mlx_active_at_start_bytes_, run_memory_figures.peak_mlx_active_bytes);
  run_memory_figures.mlx_held_at_end_bytes = read_mlx_held_bytes();
  run_memory_figures.mlx_held_change_bytes = static_cast<std::int64_t>(run_memory_figures.mlx_held_at_end_bytes) -
                                             static_cast<std::int64_t>(mlx_held_at_start_bytes_);
  const SystemWiredMemoryReading wired_reading = read_system_wired_memory();
  run_memory_figures.system_wired_at_end_bytes = wired_reading.wired_bytes;
  run_memory_figures.wire_limit_bytes = wired_reading.wire_limit_bytes;
  const MemoryPressureCounters pressure_counters = read_memory_pressure_counters();
  run_memory_figures.are_pressure_counters_readable =
      are_pressure_counters_readable_at_start_ && pressure_counters.is_readable;
  run_memory_figures.wire_requests_over_limit_count =
      counter_increase(wire_requests_over_limit_at_start_, pressure_counters.wire_requests_over_limit_count);
  run_memory_figures.compressed_bytes =
      counter_increase(compressor_input_bytes_at_start_, pressure_counters.compressor_input_bytes);
  run_memory_figures.swap_used_change_bytes = static_cast<std::int64_t>(pressure_counters.swap_used_bytes) -
                                              static_cast<std::int64_t>(swap_used_bytes_at_start_);
  return run_memory_figures;
}

} // namespace mlx_perf
