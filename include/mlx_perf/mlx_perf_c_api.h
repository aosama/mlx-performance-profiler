/*
 * mlx_perf C API: the measurement and Spark-style report of this repository,
 * callable from any language that speaks MLX-C (e.g. Rust via mlx-bridge).
 *
 * The caller describes its job as a callback that appends a FRESH, unevaluated
 * graph's outputs to a vector it is handed; the library calls it once per
 * pass (warm-up, 7 job-wall samples, timeline, serial), exactly as the C++
 * demo does, and writes `<output_directory>/<report_file_base>.html`.
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

/* Measures the job and writes its HTML report. `summary` may be NULL. */
int mlx_perf_profile_to_report(mlx_perf_build_graph_fn build_graph,
                               void* builder_payload,
                               const char* variant_name,
                               const char* report_title,
                               const char* output_directory,
                               const char* report_file_base,
                               mlx_perf_job_summary* summary);

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
