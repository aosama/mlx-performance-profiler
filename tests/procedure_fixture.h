#pragma once

// A synthetic procedure comparison shared by the procedure record, page and
// file tests: a "chatty" baseline that hands off to the GPU eight times per
// run and a "batched" candidate that hands off once, with round values chosen
// so every verdict is known in advance. No GPU.

#include "procedure_measurement.h"

#include <string>

namespace procedure_fixture {

inline mlx_perf::KernelDispatch launch_of(const std::string& metal_function_name,
                                          std::uint64_t threadgroup_count) {
  mlx_perf::KernelDispatch kernel_dispatch;
  kernel_dispatch.metal_function_name = metal_function_name;
  kernel_dispatch.grid = {threadgroup_count, 1, 1};
  kernel_dispatch.threads_per_threadgroup = {32, 1, 1};
  return kernel_dispatch;
}

// One run's memory: `working_set_mib` of peak MLX working set, nothing left
// behind, 3 GiB wired at the end under a 40 GiB limit, readable counters.
inline mlx_perf::RunMemoryFigures memory_of(std::uint64_t working_set_mib) {
  mlx_perf::RunMemoryFigures memory_figures;
  memory_figures.peak_mlx_active_added_bytes = working_set_mib << 20;
  memory_figures.peak_mlx_active_bytes = working_set_mib << 20;
  memory_figures.system_wired_at_end_bytes = 3ull << 30;
  memory_figures.wire_limit_bytes = 40ull << 30;
  memory_figures.are_pressure_counters_readable = true;
  return memory_figures;
}

// Busy 100 µs of each run on both sides; only the idle time differs.
inline mlx_perf::ProcedureRun run_of(double duration_microseconds, int blocking_wait_count) {
  mlx_perf::ProcedureRun procedure_run;
  procedure_run.duration_microseconds = duration_microseconds;
  procedure_run.gpu_busy_microseconds = 100.0;
  procedure_run.gpu_idle_microseconds = duration_microseconds - 100.0;
  procedure_run.command_buffer_count = blocking_wait_count;
  procedure_run.blocking_wait_count = blocking_wait_count;
  procedure_run.blocking_wait_microseconds = 10.0 * blocking_wait_count;
  return procedure_run;
}

// Baseline 400 µs per round with 8 waits, candidate 150 µs with 1 wait: the
// candidate is faster on duration and idle time in every round, and the GPU
// busy time is identical (within spread). Memory: the candidate's working
// set is 256 MiB more in every round (the batched loop holds every step).
inline mlx_perf::PairedProcedureMeasurement make_chatty_vs_batched(int round_count = 12) {
  mlx_perf::PairedProcedureMeasurement paired_measurement;
  paired_measurement.gpu_device_facts = {"Apple M5 Pro", 16};
  paired_measurement.baseline.procedure_name = "chatty";
  paired_measurement.candidate.procedure_name = "batched";
  for (int round_index = 0; round_index < round_count; ++round_index) {
    paired_measurement.baseline.rounds.push_back(run_of(400.0 + round_index, 8));
    paired_measurement.baseline.rounds.back().memory_figures = memory_of(64 + round_index);
    paired_measurement.candidate.rounds.push_back(run_of(150.0 + round_index, 1));
    paired_measurement.candidate.rounds.back().memory_figures = memory_of(320 + round_index);
  }
  paired_measurement.baseline.observed_run = run_of(400.0, 8);
  paired_measurement.candidate.observed_run = run_of(150.0, 1);
  for (int launch_index = 0; launch_index < 8; ++launch_index) {
    paired_measurement.baseline.observed_launches.push_back(launch_of("v_Add_float32", 2));
  }
  paired_measurement.candidate.observed_launches = {launch_of("v_Add_float32", 2),
                                                    launch_of("gemv_float32", 64)};
  return paired_measurement;
}

} // namespace procedure_fixture
