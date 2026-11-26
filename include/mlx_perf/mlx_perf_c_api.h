/*
 * mlx_perf C API: the measurement and Spark-style report of this repository,
 * callable from any language that speaks MLX-C (e.g. Rust via mlx-bridge).
 *
 * The caller describes its job as a callback that appends a FRESH, unevaluated
 * graph's outputs to a vector it is handed; the library calls it once per
 * graph it needs (11 times for one profile: stream discovery, warm-up, 7
 * job-wall samples, timeline, serial; more per variant in a comparison),
 * and writes `<output_directory>/<report_file_base>.html` plus the same
 * measurement as data in `<report_file_base>.json` (one record per node,
 * in execution order; schema in docs/measurement-file.md). Callers that only
 * want the data use mlx_perf_profile_to_measurements. Results come back as
 * files plus the plain summary struct, so no result arrays and no ownership
 * cross the boundary.
 *
 * Ownership never crosses the boundary: the library creates and frees the
 * vector; the callback only appends (MLX-C copies array handles on append).
 * Errors never unwind across it either: every entry point returns 0 on
 * success, non-zero on failure, with mlx_perf_last_error() describing it.
 *
 * Link exactly one MLX into the final binary: libmlx_perf.a carries only
 * this library's objects and resolves MLX/MLX-C symbols from the same
 * prebuilt prefix the caller links.
 */
#pragma once

#include "mlx/c/vector.h"
#include "mlx_perf/mlx_perf_version.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Appends a fresh graph's output arrays to `fresh_outputs`; returns 0 on
 * success. `builder_payload` is passed through untouched. */
typedef int (*mlx_perf_build_graph_fn)(mlx_vector_array fresh_outputs,
                                       void* builder_payload);

/* Headline numbers, in microseconds unless named otherwise. */
typedef struct mlx_perf_job_summary {
  double job_wall_median_us;
  double job_wall_min_us;
  double job_wall_max_us;
  double timeline_wall_us;
  double sum_of_stages_us;
  double measured_gpu_execution_us;
  double cross_stream_stall_us;
  int executed_node_count;
  int gpu_measured_node_count;
  int stream_count;
} mlx_perf_job_summary;

/* Measures the job and writes its HTML report and its measurement file
 * (`<report_file_base>.json`). `summary` may be NULL. */
int mlx_perf_profile_to_report(mlx_perf_build_graph_fn build_graph,
                               void* builder_payload,
                               const char* variant_name,
                               const char* report_title,
                               const char* output_directory,
                               const char* report_file_base,
                               mlx_perf_job_summary* summary);

/* Measures the job and writes only its measurement file,
 * `<output_directory>/<measurements_file_base>.json`: no DOT, SVG or HTML.
 * `summary` may be NULL. */
int mlx_perf_profile_to_measurements(mlx_perf_build_graph_fn build_graph,
                                     void* builder_payload,
                                     const char* variant_name,
                                     const char* output_directory,
                                     const char* measurements_file_base,
                                     mlx_perf_job_summary* summary);

/* One implementation to compare: a name for the reports (also part of its
 * file names, so it must be non-empty, contain no '/', and differ from the
 * other variant's) and its graph callback with that callback's payload. */
typedef struct mlx_perf_variant {
  const char* variant_name;
  mlx_perf_build_graph_fn build_graph;
  void* builder_payload;
} mlx_perf_variant;

/* Values of mlx_perf_metric_comparison.verdict. Lower is better. The verdict
 * is judged on the per-round differences (candidate − baseline, 12 rounds,
 * order flipped every round): other than "within spread" only when the
 * whole interval for the median difference lies below (or above) zero,
 * i.e. the candidate won (or lost) at least 10 of 12 rounds. Two identical
 * variants are called different under 4% of the time per metric. */
enum {
  MLX_PERF_VERDICT_CANDIDATE_FASTER = -1,
  MLX_PERF_VERDICT_WITHIN_SPREAD = 0,
  MLX_PERF_VERDICT_CANDIDATE_SLOWER = 1
};

/* One metric across the paired rounds, in microseconds. */
typedef struct mlx_perf_metric_comparison {
  double baseline_median_us;
  double baseline_min_us;
  double baseline_max_us;
  double baseline_lower_quartile_us;
  double baseline_upper_quartile_us;
  double candidate_median_us;
  double candidate_min_us;
  double candidate_max_us;
  double candidate_lower_quartile_us;
  double candidate_upper_quartile_us;
  double median_difference_us; /* median per-round candidate − baseline: negative is faster */
  double difference_interval_low_us;  /* interval for the median difference */
  double difference_interval_high_us;
  double difference_interval_coverage; /* e.g. 0.961 for 12 rounds */
  double sign_test_p_value;            /* exact, two-sided, non-zero rounds */
  double relative_difference;  /* median_difference_us / baseline median */
  int verdict;                 /* an MLX_PERF_VERDICT_* value */
  int rounds_candidate_faster;
  int round_count;
} mlx_perf_metric_comparison;

typedef struct mlx_perf_comparison_summary {
  mlx_perf_job_summary baseline;
  mlx_perf_job_summary candidate;
  mlx_perf_metric_comparison job_wall;        /* uninstrumented runs */
  mlx_perf_metric_comparison gpu_span;        /* whole job as one timed unit */
  mlx_perf_metric_comparison host_submission; /* inside the uninstrumented async_eval */
  double max_absolute_output_difference;
} mlx_perf_comparison_summary;

/* Measures two implementations of one job with their passes interleaved
 * (order flipped every round) and writes `<report_file_base>.html` (the
 * comparison, opening with its verdict), `<report_file_base>.json` (the
 * comparison as data) and, per variant, `<report_file_base>_<name>.html`
 * and `.json`. Fails, writing nothing, when the two jobs' outputs differ in
 * count, shape or dtype. `summary` may be NULL. */
int mlx_perf_compare_to_report(const mlx_perf_variant* baseline,
                               const mlx_perf_variant* candidate,
                               const char* report_title,
                               const char* output_directory,
                               const char* report_file_base,
                               mlx_perf_comparison_summary* summary);

/* The library version, "MAJOR.MINOR.PATCH" (MLX_PERF_VERSION_STRING of the
 * build that produced the library). Compare it with the header's macro to
 * catch a header and an archive from different releases. Static storage. */
const char* mlx_perf_version(void);

/* The last failure's message on this thread ("" if none). Valid until the
 * next mlx_perf_* call on the same thread. */
const char* mlx_perf_last_error(void);

#ifdef __cplusplus
}
#endif
