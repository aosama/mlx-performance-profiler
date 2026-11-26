#pragma once

// Procedure measurement: one concern only — measure two procedure jobs the
// way measure_pair measures two graph jobs: warm-ups, then interleaved
// rounds with the order flipped every round, so drift on the machine lands on
// both. Each run is timed whole, and the command buffers and blocking waits
// MLX issues inside it are recorded (metal_submission_recorder). One extra
// run per procedure records its kernel launches (metal_dispatch_recorder)
// and its wired memory (wired_memory_sampler), outside the judged rounds.
// Every run, judged or not, also reads its memory figures just before and
// just after it (run_memory_meter), so memory is judged on the same rounds.

#include "dispatch_occupancy.h"
#include "gpu_device_facts.h"
#include "procedure_job.h"
#include "run_memory_meter.h"
#include "sample_statistics.h"
#include "wired_memory_sample.h"

#include <string>
#include <vector>

namespace mlx_perf {

inline constexpr int PROCEDURE_ROUND_COUNT = 12;

// One timed run of a procedure.
struct ProcedureRun {
  double duration_microseconds = 0.0;  // run(), then its GPU work completing
  // Time inside the run when at least one of its command buffers was on the
  // GPU; the rest of the run the GPU had nothing of it to do.
  double gpu_busy_microseconds = 0.0;
  double gpu_idle_microseconds = 0.0;
  int command_buffer_count = 0;
  int blocking_wait_count = 0;          // evals and synchronizes that blocked
  double blocking_wait_microseconds = 0.0;
  // Read just before and just after the run, outside its timing (#46).
  RunMemoryFigures memory_figures;
};

struct ProcedureMeasurement {
  std::string procedure_name;
  std::vector<ProcedureRun> rounds;  // PROCEDURE_ROUND_COUNT, in order
  // The extra run that recorded kernel launches (not judged).
  ProcedureRun observed_run;
  std::vector<KernelDispatch> observed_launches;
  // The observed run's wired memory, sampled at its evaluation boundaries
  // and on a fine interval, then until macOS un-wired what it released (#45).
  std::vector<WiredMemorySample> observed_wired_memory;
};

struct PairedProcedureMeasurement {
  ProcedureMeasurement baseline;
  ProcedureMeasurement candidate;
  GpuDeviceFacts gpu_device_facts;
};

struct ProcedureComparisons {
  PairedComparison duration;
  PairedComparison gpu_busy;
  PairedComparison gpu_idle;
  // Memory, judged like time on the same rounds, in bytes; lower is better.
  PairedComparison peak_mlx_active_added;  // the run's own working set
  PairedComparison mlx_held_change;  // what a run left behind
  PairedComparison system_wired_at_end;
};

// Throws std::invalid_argument for unusable names (empty, '/', equal
// ignoring case), and std::runtime_error when this GPU's command buffers
// cannot be observed, before running anything; whatever prepare() or run()
// throws ends the measurement.
PairedProcedureMeasurement measure_procedure_pair(const ComparedProcedure& baseline,
                                                  const ComparedProcedure& candidate);

ProcedureComparisons compare_procedure_pair(const PairedProcedureMeasurement& paired_measurement);

// The per-round values of one field, in round order (for the comparison and
// its JSON record).
std::vector<double> collect_round_values(const ProcedureMeasurement& procedure_measurement,
                                         double ProcedureRun::*round_field);

// The same for a count field (command buffers, blocking waits), as doubles
// so it can be summarised like the times.
std::vector<double> collect_round_counts(const ProcedureMeasurement& procedure_measurement,
                                         int ProcedureRun::*count_field);

// The same for a memory figure, in bytes, as doubles.
std::vector<double> collect_round_bytes(const ProcedureMeasurement& procedure_measurement,
                                        std::uint64_t RunMemoryFigures::*byte_field);
std::vector<double> collect_round_bytes(const ProcedureMeasurement& procedure_measurement,
                                        std::int64_t RunMemoryFigures::*byte_field);

// Rounds whose run crossed the wire limit, and whose run saw the system
// compress memory (has_crossed_wire_limit, has_compressed_memory).
int count_rounds_crossing_wire_limit(const ProcedureMeasurement& procedure_measurement);
int count_rounds_with_compression(const ProcedureMeasurement& procedure_measurement);

} // namespace mlx_perf
