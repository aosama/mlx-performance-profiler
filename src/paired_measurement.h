#pragma once

// Paired measurement: one concern only — measure two implementations of one
// job (a baseline and a candidate) with their passes interleaved, so drift on
// a machine that is also doing other work lands on both, and keep per-round
// numbers from which a difference can be judged against the spread.
//
// The schedule: both warm-ups (whose outputs must agree), then
// PAIRED_ROUND_COUNT uninstrumented rounds (job wall and host submission)
// and PAIRED_ROUND_COUNT GPU-span rounds (the whole job as one timed unit),
// each round running both variants with the order flipped every round
// (A B, B A, …); then one per-node timeline each, for the picture only, and
// one serial pass each. With a fixed order, whichever ran second would
// always see a slightly warmer GPU; an even round count with the order
// flipped cancels that exactly.

#include "measurement_session.h"
#include "output_agreement.h"
#include "run_memory_meter.h"
#include "sample_statistics.h"

#include "mlx/stream.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_perf {

// Twelve rounds per metric: the fewest at which "won at least 10 of 12" is a
// result at under 4% false positives while two disturbed rounds still cannot
// veto it (sample_statistics.h). Even, so each variant runs first in exactly
// half the rounds. A round costs one run of the job per variant.
inline constexpr int PAIRED_ROUND_COUNT = 12;

// One implementation to compare; `graph_factory` must outlive the call.
struct ComparedJob {
  std::string variant_name;
  const GraphFactory& graph_factory;
};

// One variant's headline numbers, one entry per round, in round order. None
// comes from the per-node instrumented pass.
struct VariantRounds {
  std::vector<double> job_wall_microseconds;
  // Host time inside the single async_eval of the uninstrumented run.
  std::vector<double> host_submission_microseconds;
  // The whole job's GPU span (measure_gpu_span_once).
  std::vector<double> gpu_span_microseconds;
  // Memory around each uninstrumented run, read just before and just after
  // it, outside its timing (#46).
  std::vector<RunMemoryFigures> memory_figures;
};

// Each variant's VariantMeasurement holds one per-node timeline for the
// picture (DAG, Gantt, task table), measured after the rounds; the verdict
// never reads it. Move-only, as VariantMeasurement is.
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
  PairedComparison gpu_span;
  PairedComparison host_submission;
  // Memory, judged on the uninstrumented rounds, in bytes; lower is better.
  // Unmeasured (round_count 0) when the rounds carry no memory figures.
  PairedComparison peak_mlx_active_added;
  PairedComparison mlx_held_change;
  PairedComparison system_wired_at_end;
};

// `stream_index_to_stream` must hold every stream either job uses. Throws
// std::invalid_argument when the two jobs' outputs differ in count, shape
// or dtype (see output_agreement), before any timed pass runs.
// `measurement_observer`, when given, is told as each phase begins, under
// the label "<baseline> vs <candidate>".
PairedMeasurement measure_pair(
    const ComparedJob& baseline_job, const ComparedJob& candidate_job,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream,
    const TimelineOptions& timeline_options = {},
    MeasurementObserver* measurement_observer = nullptr);

PairComparisons compare_pair(const PairedMeasurement& paired_measurement);

} // namespace mlx_perf
