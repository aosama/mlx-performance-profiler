#include "paired_measurement.h"

#include "progress_reporter.h"

#include <array>
#include <cstdio>

namespace mlx_perf {
namespace {

// Two warm-ups, both variants in every round of two round kinds, then one
// illustrative timeline and one serial pass each.
constexpr int PAIRED_STEP_COUNT = 2 + 2 * PAIRED_ROUND_COUNT + 2 * PAIRED_ROUND_COUNT + 2 + 2;

// Everything one variant accumulates while the schedule runs.
struct PairSide {
  PairSide(const ComparedJob& compared_job, VariantMeasurement& variant_measurement,
           VariantRounds& variant_rounds)
      : timed_factory(compared_job.graph_factory),
        variant_measurement(variant_measurement),
        variant_rounds(variant_rounds) {
    variant_measurement.variant_name = compared_job.variant_name;
  }
  TimedGraphFactory timed_factory;
  VariantMeasurement& variant_measurement;
  VariantRounds& variant_rounds;
};

// Where the schedule is, for the ETA line.
struct ScheduleProgress {
  ProgressReporter progress{"paired passes", PAIRED_STEP_COUNT};
  int completed_steps = 0;
  void step_done() { progress.update(++completed_steps); }
};

// Even rounds run the baseline first, odd rounds the candidate first.
std::array<PairSide*, 2> order_for_round(int round_index, PairSide& baseline_side,
                                         PairSide& candidate_side) {
  if (round_index % 2 == 0) {
    return {&baseline_side, &candidate_side};
  }
  return {&candidate_side, &baseline_side};
}

void begin_pass_on_both(PairSide& baseline_side, PairSide& candidate_side,
                        const char* pass_name, const char* phase_description) {
  std::printf("[pair] %s...\n", phase_description);
  std::fflush(stdout);  // stay live when stdout is a pipe
  baseline_side.timed_factory.begin_pass(pass_name);
  candidate_side.timed_factory.begin_pass(pass_name);
}

OutputAgreement run_warm_ups(PairSide& baseline_side, PairSide& candidate_side,
                             ScheduleProgress& schedule_progress) {
  begin_pass_on_both(baseline_side, candidate_side, "warm_up",
                     "warm-up, both variants (compile/JIT; discarded)");
  const auto baseline_outputs = run_warm_up(baseline_side.timed_factory);
  schedule_progress.step_done();
  const auto candidate_outputs = run_warm_up(candidate_side.timed_factory);
  schedule_progress.step_done();
  return compare_outputs(baseline_outputs, candidate_outputs);
}

void run_uninstrumented_rounds(PairSide& baseline_side, PairSide& candidate_side,
                               ScheduleProgress& schedule_progress) {
  begin_pass_on_both(baseline_side, candidate_side, "job_wall",
                     "uninstrumented rounds: job wall and host submission, alternating");
  for (int round_index = 0; round_index < PAIRED_ROUND_COUNT; ++round_index) {
    for (PairSide* pair_side : order_for_round(round_index, baseline_side, candidate_side)) {
      const PipelinePassTiming pipeline_timing = sample_job_wall_once(pair_side->timed_factory);
      pair_side->variant_rounds.job_wall_microseconds.push_back(
          pipeline_timing.job_wall_microseconds);
      pair_side->variant_rounds.host_submission_microseconds.push_back(
          pipeline_timing.host_submission_microseconds);
      schedule_progress.step_done();
    }
  }
  for (PairSide* pair_side : {&baseline_side, &candidate_side}) {
    record_job_wall_samples(pair_side->variant_rounds.job_wall_microseconds,
                            pair_side->variant_measurement.job_totals);
  }
}

void run_gpu_span_rounds(PairSide& baseline_side, PairSide& candidate_side,
                         const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream,
                         ScheduleProgress& schedule_progress) {
  begin_pass_on_both(baseline_side, candidate_side, "gpu_span",
                     "GPU-span rounds: whole job as one timed unit, alternating");
  for (int round_index = 0; round_index < PAIRED_ROUND_COUNT; ++round_index) {
    for (PairSide* pair_side : order_for_round(round_index, baseline_side, candidate_side)) {
      pair_side->variant_rounds.gpu_span_microseconds.push_back(
          measure_gpu_span_once(pair_side->timed_factory, stream_index_to_stream));
      schedule_progress.step_done();
    }
  }
}

// The per-node picture: one timeline per variant, measured after the rounds
// and never read by the verdict.
void run_illustrative_timelines(
    PairSide& baseline_side, PairSide& candidate_side,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream,
    const TimelineOptions& timeline_options, ScheduleProgress& schedule_progress) {
  begin_pass_on_both(baseline_side, candidate_side, "timeline",
                     "timeline per variant (the picture; not part of the verdict)");
  for (PairSide* pair_side : {&baseline_side, &candidate_side}) {
    TimelineCapture timeline_capture =
        capture_timeline(pair_side->timed_factory, stream_index_to_stream, timeline_options);
    pair_side->variant_measurement.timeline_graph = std::move(timeline_capture.timeline_graph);
    pair_side->variant_measurement.gpu_timeline = std::move(timeline_capture.gpu_timeline);
    schedule_progress.step_done();
  }
}

void run_serial_passes(PairSide& baseline_side, PairSide& candidate_side,
                       const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream,
                       ScheduleProgress& schedule_progress) {
  begin_pass_on_both(baseline_side, candidate_side, "serial",
                     "serial pass, each variant (sum of stages)");
  for (PairSide* pair_side : {&baseline_side, &candidate_side}) {
    run_serial_pass(pair_side->timed_factory, stream_index_to_stream,
                    pair_side->variant_measurement);
    pair_side->variant_measurement.graph_build_passes = pair_side->timed_factory.build_passes();
    schedule_progress.step_done();
  }
}

} // namespace

PairedMeasurement measure_pair(
    const ComparedJob& baseline_job, const ComparedJob& candidate_job,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream,
    const TimelineOptions& timeline_options) {
  PairedMeasurement paired_measurement;
  PairSide baseline_side(baseline_job, paired_measurement.baseline,
                         paired_measurement.baseline_rounds);
  PairSide candidate_side(candidate_job, paired_measurement.candidate,
                          paired_measurement.candidate_rounds);
  ScheduleProgress schedule_progress;
  paired_measurement.output_agreement =
      run_warm_ups(baseline_side, candidate_side, schedule_progress);
  run_uninstrumented_rounds(baseline_side, candidate_side, schedule_progress);
  run_gpu_span_rounds(baseline_side, candidate_side, stream_index_to_stream, schedule_progress);
  run_illustrative_timelines(baseline_side, candidate_side, stream_index_to_stream,
                             timeline_options, schedule_progress);
  run_serial_passes(baseline_side, candidate_side, stream_index_to_stream, schedule_progress);
  schedule_progress.progress.finish(schedule_progress.completed_steps);
  return paired_measurement;
}

PairComparisons compare_pair(const PairedMeasurement& paired_measurement) {
  const VariantRounds& baseline_rounds = paired_measurement.baseline_rounds;
  const VariantRounds& candidate_rounds = paired_measurement.candidate_rounds;
  return {compare_paired_samples(baseline_rounds.job_wall_microseconds,
                                 candidate_rounds.job_wall_microseconds),
          compare_paired_samples(baseline_rounds.gpu_span_microseconds,
                                 candidate_rounds.gpu_span_microseconds),
          compare_paired_samples(baseline_rounds.host_submission_microseconds,
                                 candidate_rounds.host_submission_microseconds)};
}

} // namespace mlx_perf
