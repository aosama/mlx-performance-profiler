#pragma once

// Paired measurement: one concern only — measure two implementations of one
// job (a baseline and a candidate) with their passes interleaved, so drift on
// a machine that is also doing other work lands on both, and keep per-round
// numbers from which a difference can be judged against the spread.
//
// The schedule: both warm-ups (whose outputs must agree), then
// JOB_WALL_SAMPLE_COUNT job-wall rounds and PAIRED_TIMELINE_ROUND_COUNT
// timeline rounds, each round running both variants with the order flipped
// every round (A B, B A, …), then one serial pass each. With a fixed order,
// whichever ran second would always see a slightly warmer GPU; flipping
// cancels that along with slow drift.

#include "measurement_session.h"
#include "output_agreement.h"
#include "sample_statistics.h"

#include "mlx/stream.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_perf {

// Five timeline rounds per variant: GPU execution gets a median and a range
// of its own. More rounds would hold more timelines in memory (each round's
// graph stays alive until the median round is chosen).
inline constexpr int PAIRED_TIMELINE_ROUND_COUNT = 5;

// One implementation to compare; `graph_factory` must outlive the call.
struct ComparedJob {
  std::string variant_name;
  const GraphFactory& graph_factory;
};

// One variant's headline numbers, one entry per round, in round order.
struct VariantRounds {
  std::vector<double> job_wall_microseconds;
  std::vector<double> gpu_execution_microseconds;
  std::vector<double> timeline_wall_microseconds;
  // Host time inside async_eval for the whole job: encoding and committing.
  std::vector<double> host_submission_microseconds;
};

// Each variant's VariantMeasurement holds the timeline round whose GPU
// execution was the median, so the picture shown is the run the headline
// number came from. Move-only, as VariantMeasurement is.
struct PairedMeasurement {
  VariantMeasurement baseline;
  VariantMeasurement candidate;
  VariantRounds baseline_rounds;
  VariantRounds candidate_rounds;
  OutputAgreement output_agreement;
};

// The three questions a comparison answers, each against its own spread.
struct PairComparisons {
  PairedComparison job_wall;
  PairedComparison gpu_execution;
  PairedComparison host_submission;
};

// `stream_index_to_stream` must hold every stream either job uses. Throws
// std::invalid_argument when the two jobs' outputs differ in count, shape
// or dtype (see output_agreement), before any timed pass runs.
PairedMeasurement measure_pair(
    const ComparedJob& baseline_job, const ComparedJob& candidate_job,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream,
    const TimelineOptions& timeline_options = {});

PairComparisons compare_pair(const PairedMeasurement& paired_measurement);

} // namespace mlx_perf
