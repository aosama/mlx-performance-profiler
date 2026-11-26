#include "procedure_measurement.h"

#include "console_output.h"
#include "metal_dispatch_recorder.h"
#include "metal_host_clock.h"
#include "metal_submission_recorder.h"
#include "progress_reporter.h"
#include "wired_memory_sampler.h"
#include "text_format.h"

#include "mlx/transforms.h"

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace mlx_perf {
namespace {

// Per procedure: one warm-up, one observed run, then one run per round.
constexpr int PROCEDURE_RUN_COUNT = 2 + 2 + 2 * PROCEDURE_ROUND_COUNT;

void require_usable_names(const ComparedProcedure& baseline, const ComparedProcedure& candidate) {
  for (const ComparedProcedure* compared_procedure : {&baseline, &candidate}) {
    const std::string& procedure_name = compared_procedure->procedure_name;
    if (procedure_name.empty() || procedure_name.find('/') != std::string::npos) {
      throw std::invalid_argument("procedure name \"" + procedure_name + "\" must be non-empty without '/'");
    }
  }
  // On macOS's default file system "Read" and "read" name the same files.
  if (are_equal_ignoring_case(baseline.procedure_name, candidate.procedure_name)) {
    throw std::invalid_argument("both procedures are named \"" + baseline.procedure_name + "\" (ignoring case)");
  }
}

// The later of run() returning and its last command buffer finishing on the
// GPU. Read from the timestamps rather than the clock after the profiler's
// own wait, whose per-buffer bookkeeping (status, timestamps, release) would
// otherwise be charged to the procedure, more so the more buffers it commits.
double find_run_finished_seconds(double run_returned_seconds, const std::vector<GpuWindow>& command_buffer_windows) {
  double run_finished_seconds = run_returned_seconds;
  for (const GpuWindow& command_buffer_window : command_buffer_windows) {
    run_finished_seconds = std::max(run_finished_seconds, command_buffer_window.finished_seconds);
  }
  return run_finished_seconds;
}

// What the observed run records beside its timing; the judged rounds
// record neither, so their timing carries no observer's cost.
struct ObservedRunRecords {
  std::vector<KernelDispatch> kernel_launches;
  std::vector<WiredMemorySample> wired_memory_samples;
};

ProcedureRun summarize_run(const SubmissionRecord& submission_record, double started_seconds,
                           double finished_seconds) {
  ProcedureRun procedure_run;
  procedure_run.duration_microseconds = (finished_seconds - started_seconds) * 1e6;
  procedure_run.gpu_busy_microseconds = measure_gpu_busy_microseconds(
      submission_record.command_buffer_windows, started_seconds, finished_seconds);
  procedure_run.gpu_idle_microseconds =
      procedure_run.duration_microseconds - procedure_run.gpu_busy_microseconds;
  procedure_run.command_buffer_count = submission_record.committed_command_buffer_count;
  procedure_run.blocking_wait_count = submission_record.blocking_wait_count;
  procedure_run.blocking_wait_microseconds = submission_record.blocking_wait_microseconds;
  return procedure_run;
}

// prepare() and its work finished, untimed; then run() timed until every
// command buffer it committed has completed. The profiler's own wait for
// those buffers is not counted as one of the procedure's blocking waits.
// Whole-process recording: run() may hand its MLX work to a worker thread,
// and only one procedure runs at a time, so everything submitted meanwhile
// is its work (#44).
ProcedureRun run_procedure_once(ProcedureJob& procedure_job, ObservedRunRecords* observed_run_records) {
  procedure_job.prepare();
  mlx::core::synchronize();
  const bool is_observed = observed_run_records != nullptr;
  std::unique_ptr<DispatchRecordingScope> dispatch_scope =
      is_observed ? std::make_unique<DispatchRecordingScope>(RecordingReach::whole_process) : nullptr;
  std::unique_ptr<WiredMemorySampler> wired_memory_sampler =
      is_observed ? std::make_unique<WiredMemorySampler>() : nullptr;
  SubmissionRecordingScope submission_scope(RecordingReach::whole_process);
  if (is_observed) {
    submission_scope.set_evaluation_boundary_listener(wired_memory_sampler.get());
    wired_memory_sampler->start();
  }
  const double started_seconds = current_metal_host_seconds();
  procedure_job.run();
  const double run_returned_seconds = current_metal_host_seconds();
  const SubmissionRecord submission_record = submission_scope.finish_and_take_record();
  const double finished_seconds =
      find_run_finished_seconds(run_returned_seconds, submission_record.command_buffer_windows);
  if (is_observed) {
    submission_scope.set_evaluation_boundary_listener(nullptr);
    observed_run_records->kernel_launches = dispatch_scope->take_recorded_dispatches();
    observed_run_records->wired_memory_samples = wired_memory_sampler->finish();
  }
  return summarize_run(submission_record, started_seconds, finished_seconds);
}

struct PairProgress {
  ProgressReporter progress_reporter{"procedure runs", PROCEDURE_RUN_COUNT};
  int completed_run_count = 0;
  void advance() { progress_reporter.update(++completed_run_count); }
};

void run_warm_ups_and_observed_runs(const ComparedProcedure& baseline, const ComparedProcedure& candidate,
                                    PairedProcedureMeasurement& paired_measurement, PairProgress& pair_progress) {
  print_progress_line("[procedures] warm-up: %s, %s\n", baseline.procedure_name.c_str(),
                      candidate.procedure_name.c_str());
  for (const ComparedProcedure* compared_procedure : {&baseline, &candidate}) {
    run_procedure_once(compared_procedure->procedure_job, nullptr);
    pair_progress.advance();
  }
  print_progress_line("[procedures] observed run of each: kernel launches and wired memory\n");
  for (ProcedureMeasurement* procedure_measurement : {&paired_measurement.baseline, &paired_measurement.candidate}) {
    const ComparedProcedure& compared_procedure =
        procedure_measurement == &paired_measurement.baseline ? baseline : candidate;
    ObservedRunRecords observed_run_records;
    procedure_measurement->observed_run = run_procedure_once(compared_procedure.procedure_job, &observed_run_records);
    procedure_measurement->observed_launches = std::move(observed_run_records.kernel_launches);
    procedure_measurement->observed_wired_memory = std::move(observed_run_records.wired_memory_samples);
    pair_progress.advance();
  }
}

} // namespace

PairedProcedureMeasurement measure_procedure_pair(const ComparedProcedure& baseline,
                                                  const ComparedProcedure& candidate) {
  require_usable_names(baseline, candidate);
  // Without the hooks every run would read as 0 command buffers and 100% GPU
  // idle: a confident wrong answer, so refuse instead.
  if (!install_submission_recorder()) {
    throw std::runtime_error("cannot observe this GPU's Metal command buffers: procedures cannot be measured");
  }
  install_dispatch_recorder();
  PairedProcedureMeasurement paired_measurement;
  paired_measurement.baseline.procedure_name = baseline.procedure_name;
  paired_measurement.candidate.procedure_name = candidate.procedure_name;
  paired_measurement.gpu_device_facts = read_gpu_device_facts();
  PairProgress pair_progress;
  run_warm_ups_and_observed_runs(baseline, candidate, paired_measurement, pair_progress);
  print_progress_line("[procedures] %d interleaved rounds, flipping which goes first\n", PROCEDURE_ROUND_COUNT);
  for (int round_index = 0; round_index < PROCEDURE_ROUND_COUNT; ++round_index) {
    const bool is_baseline_first = round_index % 2 == 0;
    const ComparedProcedure& first_procedure = is_baseline_first ? baseline : candidate;
    const ComparedProcedure& second_procedure = is_baseline_first ? candidate : baseline;
    ProcedureMeasurement& first_measurement =
        is_baseline_first ? paired_measurement.baseline : paired_measurement.candidate;
    ProcedureMeasurement& second_measurement =
        is_baseline_first ? paired_measurement.candidate : paired_measurement.baseline;
    first_measurement.rounds.push_back(run_procedure_once(first_procedure.procedure_job, nullptr));
    pair_progress.advance();
    second_measurement.rounds.push_back(run_procedure_once(second_procedure.procedure_job, nullptr));
    pair_progress.advance();
  }
  pair_progress.progress_reporter.finish(pair_progress.completed_run_count);
  return paired_measurement;
}

std::vector<double> collect_round_values(const ProcedureMeasurement& procedure_measurement,
                                         double ProcedureRun::*round_field) {
  std::vector<double> round_values;
  for (const ProcedureRun& procedure_run : procedure_measurement.rounds) {
    round_values.push_back(procedure_run.*round_field);
  }
  return round_values;
}

std::vector<double> collect_round_counts(const ProcedureMeasurement& procedure_measurement,
                                         int ProcedureRun::*count_field) {
  std::vector<double> round_counts;
  for (const ProcedureRun& procedure_run : procedure_measurement.rounds) {
    round_counts.push_back(procedure_run.*count_field);
  }
  return round_counts;
}

ProcedureComparisons compare_procedure_pair(const PairedProcedureMeasurement& paired_measurement) {
  const ProcedureMeasurement& baseline = paired_measurement.baseline;
  const ProcedureMeasurement& candidate = paired_measurement.candidate;
  return ProcedureComparisons{
      compare_paired_samples(collect_round_values(baseline, &ProcedureRun::duration_microseconds),
                             collect_round_values(candidate, &ProcedureRun::duration_microseconds)),
      compare_paired_samples(collect_round_values(baseline, &ProcedureRun::gpu_busy_microseconds),
                             collect_round_values(candidate, &ProcedureRun::gpu_busy_microseconds)),
      compare_paired_samples(collect_round_values(baseline, &ProcedureRun::gpu_idle_microseconds),
                             collect_round_values(candidate, &ProcedureRun::gpu_idle_microseconds))};
}

} // namespace mlx_perf
