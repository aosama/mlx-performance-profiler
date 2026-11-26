#include "paired_measurement.h"

#include "progress_reporter.h"
#include "timed_units.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <utility>

namespace mlx_perf {
namespace {

// Two warm-ups, two serial passes, and both variants in every round.
constexpr int PAIRED_STEP_COUNT =
    2 + 2 * JOB_WALL_SAMPLE_COUNT + 2 * PAIRED_TIMELINE_ROUND_COUNT + 2;

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
  std::vector<TimelineCapture> timeline_captures;
};

// Even rounds run the baseline first, odd rounds the candidate first.
std::array<PairSide*, 2> order_for_round(int round_index, PairSide& baseline_side,
                                         PairSide& candidate_side) {
  if (round_index % 2 == 0) {
    return {&baseline_side, &candidate_side};
  }
  return {&candidate_side, &baseline_side};
}

void announce_phase(const char* phase_description) {
  std::printf("[pair] %s...\n", phase_description);
  std::fflush(stdout);  // stay live when stdout is a pipe
}

OutputAgreement run_warm_ups(PairSide& baseline_side, PairSide& candidate_side,
                             int& completed_steps, ProgressReporter& progress) {
  announce_phase("warm-up, both variants (compile/JIT; discarded)");
  baseline_side.timed_factory.begin_pass("warm_up");
  candidate_side.timed_factory.begin_pass("warm_up");
  const auto baseline_outputs = run_warm_up(baseline_side.timed_factory);
  const auto candidate_outputs = run_warm_up(candidate_side.timed_factory);
  completed_steps += 2;
  progress.update(completed_steps);
  return compare_outputs(baseline_outputs, candidate_outputs);
}

void run_job_wall_rounds(PairSide& baseline_side, PairSide& candidate_side,
                         int& completed_steps, ProgressReporter& progress) {
  announce_phase("job-wall rounds, alternating");
  baseline_side.timed_factory.begin_pass("job_wall");
  candidate_side.timed_factory.begin_pass("job_wall");
  for (int round_index = 0; round_index < JOB_WALL_SAMPLE_COUNT; ++round_index) {
    for (PairSide* pair_side : order_for_round(round_index, baseline_side, candidate_side)) {
      pair_side->variant_rounds.job_wall_microseconds.push_back(
          sample_job_wall_once(pair_side->timed_factory));
      progress.update(++completed_steps);
    }
  }
  for (PairSide* pair_side : {&baseline_side, &candidate_side}) {
    record_job_wall_samples(pair_side->variant_rounds.job_wall_microseconds,
                            pair_side->variant_measurement.job_totals);
  }
}

void record_timeline_round(PairSide& pair_side, TimelineCapture timeline_capture) {
  const GpuTimeline& gpu_timeline = timeline_capture.gpu_timeline;
  VariantRounds& variant_rounds = pair_side.variant_rounds;
  variant_rounds.gpu_execution_microseconds.push_back(
      gpu_timeline.measured_gpu_compute_microseconds);
  variant_rounds.timeline_wall_microseconds.push_back(gpu_timeline.timeline_wall_microseconds);
  variant_rounds.host_submission_microseconds.push_back(
      sum_host_submission_microseconds(collect_timed_units(gpu_timeline)));
  pair_side.timeline_captures.push_back(std::move(timeline_capture));
}

// Keeps the round whose GPU execution is the median (the lower middle for an
// even count) and releases the others' graphs.
void keep_median_timeline(PairSide& pair_side) {
  const std::vector<double>& gpu_execution = pair_side.variant_rounds.gpu_execution_microseconds;
  // (GPU execution, round) pairs sort by time, then by round for ties.
  std::vector<std::pair<double, std::size_t>> gpu_execution_by_round;
  for (std::size_t round_index = 0; round_index < gpu_execution.size(); ++round_index) {
    gpu_execution_by_round.emplace_back(gpu_execution[round_index], round_index);
  }
  std::sort(gpu_execution_by_round.begin(), gpu_execution_by_round.end());
  const std::size_t median_round =
      gpu_execution_by_round[(gpu_execution_by_round.size() - 1) / 2].second;
  TimelineCapture& median_capture = pair_side.timeline_captures[median_round];
  pair_side.variant_measurement.timeline_graph = std::move(median_capture.timeline_graph);
  pair_side.variant_measurement.gpu_timeline = std::move(median_capture.gpu_timeline);
  pair_side.timeline_captures.clear();
}

void run_timeline_rounds(PairSide& baseline_side, PairSide& candidate_side,
                         const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream,
                         const TimelineOptions& timeline_options, int& completed_steps,
                         ProgressReporter& progress) {
  announce_phase("timeline rounds, alternating (overlap kept)");
  baseline_side.timed_factory.begin_pass("timeline");
  candidate_side.timed_factory.begin_pass("timeline");
  for (int round_index = 0; round_index < PAIRED_TIMELINE_ROUND_COUNT; ++round_index) {
    for (PairSide* pair_side : order_for_round(round_index, baseline_side, candidate_side)) {
      record_timeline_round(*pair_side, capture_timeline(pair_side->timed_factory,
                                                         stream_index_to_stream,
                                                         timeline_options));
      progress.update(++completed_steps);
    }
  }
  keep_median_timeline(baseline_side);
  keep_median_timeline(candidate_side);
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
  ProgressReporter progress("paired passes", PAIRED_STEP_COUNT);
  int completed_steps = 0;
  paired_measurement.output_agreement =
      run_warm_ups(baseline_side, candidate_side, completed_steps, progress);
  run_job_wall_rounds(baseline_side, candidate_side, completed_steps, progress);
  run_timeline_rounds(baseline_side, candidate_side, stream_index_to_stream, timeline_options,
                      completed_steps, progress);
  announce_phase("serial pass, each variant (sum of stages)");
  for (PairSide* pair_side : {&baseline_side, &candidate_side}) {
    pair_side->timed_factory.begin_pass("serial");
    run_serial_pass(pair_side->timed_factory, stream_index_to_stream,
                    pair_side->variant_measurement);
    progress.update(++completed_steps);
    pair_side->variant_measurement.graph_build_passes = pair_side->timed_factory.build_passes();
  }
  progress.finish(completed_steps);
  return paired_measurement;
}

PairComparisons compare_pair(const PairedMeasurement& paired_measurement) {
  const VariantRounds& baseline_rounds = paired_measurement.baseline_rounds;
  const VariantRounds& candidate_rounds = paired_measurement.candidate_rounds;
  return {compare_paired_samples(baseline_rounds.job_wall_microseconds,
                                 candidate_rounds.job_wall_microseconds),
          compare_paired_samples(baseline_rounds.gpu_execution_microseconds,
                                 candidate_rounds.gpu_execution_microseconds),
          compare_paired_samples(baseline_rounds.host_submission_microseconds,
                                 candidate_rounds.host_submission_microseconds)};
}

} // namespace mlx_perf
