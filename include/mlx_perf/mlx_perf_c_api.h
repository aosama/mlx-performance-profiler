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

#include <stdint.h>

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

/* One step of a procedure: returns 0 on success. Any other value ends the
 * comparison, writing nothing, with "procedure <name>: run returned <n>"
 * (or "prepare returned <n>") in mlx_perf_last_error(). */
typedef int (*mlx_perf_procedure_step_fn)(void* procedure_payload);

/* A procedure: any code that does MLX work, however many graphs and
 * evaluations it makes (a decode loop, a pipeline with host logic between
 * evaluations). `run` is measured whole, from its start until every command
 * buffer it committed has completed. `prepare` (may be NULL) runs before
 * every run, untimed: put setup there (fresh inputs, a reset cache), and
 * evaluate what it creates before returning, so that work is not left to the
 * timed run. Both are called on the calling thread; `run` may hand its MLX
 * work to other threads (an engine's owner thread), because every command
 * buffer, wait and launch anywhere in the process while it runs is counted
 * as its own, so use MLX for nothing else meanwhile. The name
 * follows the rules of mlx_perf_variant.variant_name. */
typedef struct mlx_perf_procedure {
  const char* procedure_name;
  mlx_perf_procedure_step_fn prepare;
  mlx_perf_procedure_step_fn run;
  void* procedure_payload;
} mlx_perf_procedure;

/* One procedure's hand-offs: medians over the 12 rounds, and its kernel
 * launches from one extra observed run. */
typedef struct mlx_perf_procedure_summary {
  double median_command_buffer_count;  /* MLX's submissions to the GPU */
  double median_blocking_wait_count;   /* evals and synchronizes that blocked */
  double median_blocking_wait_us;      /* host time spent blocked */
  int kernel_launch_count;
  /* Launches with fewer threadgroups than the GPU has cores; -1 when the
   * core count is unknown. */
  int launches_unable_to_fill_gpu;
} mlx_perf_procedure_summary;

/* One memory figure across the paired rounds, in bytes, judged like time:
 * lower is better. `verdict` is an MLX_PERF_VERDICT_* value, where
 * CANDIDATE_FASTER means the candidate used less. */
typedef struct mlx_perf_memory_comparison {
  double baseline_median_bytes;
  double candidate_median_bytes;
  double median_difference_bytes; /* median per-round candidate − baseline: negative is less */
  double difference_interval_low_bytes;
  double difference_interval_high_bytes;
  double relative_difference;     /* median_difference_bytes / baseline median */
  int verdict;
  int rounds_candidate_less;
  int round_count;
} mlx_perf_memory_comparison;

typedef struct mlx_perf_procedure_comparison_summary {
  mlx_perf_procedure_summary baseline;
  mlx_perf_procedure_summary candidate;
  mlx_perf_metric_comparison duration; /* run() until its GPU work completed */
  mlx_perf_metric_comparison gpu_busy; /* at least one of its buffers on the GPU */
  mlx_perf_metric_comparison gpu_idle; /* the rest: the GPU waiting on the host */
  /* Memory on the same rounds, read just before and after each run (#46). */
  mlx_perf_memory_comparison peak_mlx_active_added; /* MLX peak active less active at the start */
  mlx_perf_memory_comparison mlx_held_change;          /* MLX active + cache, end less start */
  mlx_perf_memory_comparison system_wired_at_end;
  /* Rounds whose run crossed the wire limit (the kernel refused wire requests
   * over it, or the wired total ended over it), and rounds in which the system
   * compressed memory (any program's). */
  int baseline_rounds_crossing_wire_limit;
  int candidate_rounds_crossing_wire_limit;
  int baseline_rounds_with_compression;
  int candidate_rounds_with_compression;
} mlx_perf_procedure_comparison_summary;

/* Measures two procedures the way mlx_perf_compare_to_report measures two
 * jobs (one warm-up and one observed run each, then 12 rounds, order flipped
 * every round) and writes `<report_file_base>.html` (the verdict, the
 * hand-offs per run, and the kernel launches by Metal function) and
 * `<report_file_base>.json` (`"kind": "procedure_comparison"`). Use it when
 * an implementation is more than one graph; for one graph,
 * mlx_perf_compare_to_report shows more. `summary` may be NULL. */
int mlx_perf_compare_procedures_to_report(const mlx_perf_procedure* baseline,
                                          const mlx_perf_procedure* candidate,
                                          const char* report_title,
                                          const char* output_directory,
                                          const char* report_file_base,
                                          mlx_perf_procedure_comparison_summary* summary);

/* One instant's wired memory: the system's wired total and the line macOS
 * keeps it under (vm.global_user_wire_limit), beside MLX's own holding
 * (active arrays plus buffer cache). On Apple silicon the GPU's memory is
 * wired, so `wire_headroom_bytes` is the room left before macOS stops keeping
 * more of it wired. `wired_outside_mlx_bytes` is the figure an engine sizes
 * against. macOS wires memory when the GPU uses it, un-wires memory MLX
 * released a fraction of a second later (see
 * mlx_perf_measure_wire_release_lag), and un-wires memory MLX still holds
 * after about a second without GPU use, wiring it again on its next use. So
 * that figure is too high just after a release and too low while MLX's
 * memory sits idle. */
typedef struct mlx_perf_wired_memory {
  uint64_t system_wired_bytes;
  uint64_t wire_limit_bytes;      /* 0 when unknown */
  uint64_t mlx_active_bytes;
  uint64_t mlx_cache_bytes;
  int64_t wired_outside_mlx_bytes; /* system wired − (MLX active + cache) */
  int64_t wire_headroom_bytes;     /* limit − system wired; 0 when the limit is unknown */
} mlx_perf_wired_memory;

/* Reads the wired memory now: two system calls and MLX's two counters, cheap
 * enough to call after every evaluation. Fails when the kernel refuses the
 * read. Thread-safe. */
int mlx_perf_read_wired_memory(mlx_perf_wired_memory* reading);

/* How long macOS takes, on this Mac, to un-wire memory after MLX releases
 * it. Lags are NaN when no repetition resolved. */
typedef struct mlx_perf_wire_release_lag {
  uint64_t allocation_bytes;     /* as measured: rounded down to float32 elements */
  uint64_t wire_limit_bytes;     /* 0 when unknown */
  int64_t median_wired_rise_bytes; /* how far the system's wired total rose per allocation */
  double median_release_lag_us;
  double largest_release_lag_us;
  int repetition_count;
  int resolved_count;            /* repetitions whose un-wiring was seen in time */
} mlx_perf_wire_release_lag;

/* Allocates and evaluates `allocation_bytes` (under 8 GiB) on the GPU,
 * releases it and clears MLX's cache, then times from MLX's holding falling
 * to the system's wired total falling by at least half the allocation;
 * `repetition_count` times (a fraction of a second each). Each
 * repetition first waits for the wired total to settle (at most 3 s) and
 * then for the lag, so the library prints progress with an ETA. Run it with
 * nothing else using MLX. */
int mlx_perf_measure_wire_release_lag(uint64_t allocation_bytes, int repetition_count,
                                      mlx_perf_wire_release_lag* result);

/* Switches the library's console output on (non-zero, the default) or off
 * (0): progress and file lines on stdout, error lines on stderr. Off, the
 * library writes nothing to either; failures still come back as return
 * codes and mlx_perf_last_error(). Process-wide and thread-safe. */
void mlx_perf_set_console_output(int is_enabled);

/* Optional: starts observing the kernel launches MLX encodes, so that every
 * kernel MLX compiles from now on is named in the launch records
 * (`kernel_dispatches[].metal_function_name` in the measurement file). Call
 * it before your program's first MLX evaluation. Without it the first
 * profile starts the observer, and kernels compiled earlier are recorded
 * unnamed (their shapes are still recorded). It never changes what MLX
 * computes. Returns 1 when observing, 0 when this GPU's Metal classes could
 * not be observed (launch records are then empty). Safe to call repeatedly. */
int mlx_perf_install_launch_recorder(void);

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
