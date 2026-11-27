# C API reference

Every declaration in `include/mlx_perf/mlx_perf_c_api.h` and `include/mlx_perf/mlx_perf_version.h`, as of 0.7.0. A test fails if this page misses a function, struct or constant from the header, or quotes a prototype that no longer matches it.

```c
#include "mlx_perf/mlx_perf_c_api.h"   /* also includes mlx/c/vector.h and mlx_perf_version.h */
```

**Conventions.** Every function that measures returns `0` on success and non-zero on failure, and never throws or unwinds into the caller. After a failure, `mlx_perf_last_error()` describes it. Times are microseconds (`_us`). No MLX array or other ownership crosses the boundary: results come back as files and plain structs.

## Functions

### mlx_perf_profile_to_report

```c
int mlx_perf_profile_to_report(mlx_perf_build_graph_fn build_graph,
                               void* builder_payload,
                               const char* variant_name,
                               const char* report_title,
                               const char* output_directory,
                               const char* report_file_base,
                               mlx_perf_job_summary* summary);
```

Measures the job `build_graph` builds and writes `<output_directory>/<report_file_base>.html` (the report), `.json` (the [measurement file](../measurement-file.md)) and `.dot`. `variant_name` names the run inside the report and the JSON. `report_title` is the report's heading. `summary` may be `NULL`. Calls `build_graph` 11 times. Guide: [Profiling a job](../profiling-a-job.md).

### mlx_perf_profile_to_measurements

```c
int mlx_perf_profile_to_measurements(mlx_perf_build_graph_fn build_graph,
                                     void* builder_payload,
                                     const char* variant_name,
                                     const char* output_directory,
                                     const char* measurements_file_base,
                                     mlx_perf_job_summary* summary);
```

The same measurement, writing only `<output_directory>/<measurements_file_base>.json`: no HTML or DOT. `summary` may be `NULL`. Guide: [Measuring from code](../measuring-from-code.md).

### mlx_perf_compare_to_report

```c
int mlx_perf_compare_to_report(const mlx_perf_variant* baseline,
                               const mlx_perf_variant* candidate,
                               const char* report_title,
                               const char* output_directory,
                               const char* report_file_base,
                               mlx_perf_comparison_summary* summary);
```

Measures two implementations of one job with their runs interleaved (12 rounds, order flipped every round) and writes `<report_file_base>.html` (the comparison, opening with its verdict), `<report_file_base>.json` (the [comparison record](../measurement-file.md#the-comparison-record-kind-comparison)) and, per variant, `<report_file_base>_<variant_name>.html` and `.json`. Fails, writing nothing, when the two jobs' outputs differ in count, shape or dtype, or the variant names are unusable. `summary` may be `NULL`. Guide: [Comparing two implementations](../comparing-implementations.md).

### mlx_perf_compare_procedures_to_report

```c
int mlx_perf_compare_procedures_to_report(const mlx_perf_procedure* baseline,
                                          const mlx_perf_procedure* candidate,
                                          const char* report_title,
                                          const char* output_directory,
                                          const char* report_file_base,
                                          mlx_perf_procedure_comparison_summary* summary);
```

Measures two procedures, code that evaluates any number of MLX graphs, each run whole: one warm-up and one observed run each, then 12 rounds with the order flipped every round. Writes `<report_file_base>.html` (the verdict, the hand-offs per run and the kernel launches by Metal function) and `<report_file_base>.json` (the [procedure comparison record](../measurement-file.md#the-procedure-comparison-record-kind-procedure_comparison)). Fails, writing nothing, when a callback returns non-zero or the names are unusable. `summary` may be `NULL`. Guide: [Comparing procedures](../comparing-procedures.md).

### mlx_perf_read_wired_memory

```c
int mlx_perf_read_wired_memory(mlx_perf_wired_memory* reading);
```

Reads the wired memory now into `reading`: the system's wired total and the wire limit, MLX's active memory and cache, and the derived "wired outside MLX" and room under the limit. Two system calls and MLX's two counters: cheap enough to call after every evaluation. Thread-safe. Fails when `reading` is `NULL` or the kernel refuses the read. Guide: [Wired memory](../wired-memory.md).

### mlx_perf_measure_wire_release_lag

```c
int mlx_perf_measure_wire_release_lag(uint64_t allocation_bytes, int repetition_count,
                                      mlx_perf_wire_release_lag* result);
```

Measures how long macOS keeps memory wired after MLX releases it, on this Mac. Each repetition waits for the wired total to settle (at most 3 s), allocates and evaluates `allocation_bytes` on the GPU, releases it and clears MLX's cache, then times from MLX's holding falling to the wired total falling by at least half the allocation (waiting at most 5 s). Prints progress with an ETA. `allocation_bytes` must be at least 4 and under 8 GiB; `repetition_count` at least 1. Run it with nothing else using MLX.

### mlx_perf_set_console_output

```c
void mlx_perf_set_console_output(int is_enabled);
```

`0` silences everything the library prints (progress and file lines on stdout, error lines on stderr); non-zero, the default, restores it. Process-wide and thread-safe. Failures still come back as return codes and through `mlx_perf_last_error()`.

### mlx_perf_install_launch_recorder

```c
int mlx_perf_install_launch_recorder(void);
```

Optional. Starts observing the kernel launches MLX encodes, so every kernel MLX compiles from then on is named in the launch records (`kernel_dispatches[].kernel_function_name` in the [measurement file](../measurement-file.md#nodes)). Call it before your program's first MLX evaluation. Without it the first profile starts the observer, and kernels compiled earlier are recorded unnamed, though their shapes are still recorded. It never changes what MLX computes. Returns `1` when observing, `0` when this GPU's Metal classes cannot be observed (launch records are then empty). Safe to call repeatedly.

### mlx_perf_version

```c
const char* mlx_perf_version(void);
```

The library's version, `"MAJOR.MINOR.PATCH"`, as built. Compare it with `MLX_PERF_VERSION_STRING` to catch a header and an archive from different releases. Static storage: never free it.

### mlx_perf_last_error

```c
const char* mlx_perf_last_error(void);
```

The last failure's message on the calling thread, or `""` if none. Valid until the next `mlx_perf_*` call on the same thread. Every message is listed in [Troubleshooting](../troubleshooting.md).

## Types

### mlx_perf_build_graph_fn

```c
typedef int (*mlx_perf_build_graph_fn)(mlx_vector_array fresh_outputs,
                                       void* builder_payload);
```

Your callback. It must append the outputs of a **fresh, unevaluated** graph to `fresh_outputs` and return `0`. `builder_payload` is passed through untouched. The library creates and frees `fresh_outputs`; MLX-C copies a handle on append. Contract in full: [The callback contract](../profiling-a-job.md#the-callback-contract).

### mlx_perf_job_summary

```c
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
```

| Field | Meaning |
|---|---|
| `job_wall_median_us`, `job_wall_min_us`, `job_wall_max_us` | the uninstrumented job duration over 7 warm runs: the number to quote |
| `timeline_wall_us` | the instrumented timeline pass, first issue to last completion; longer than the job duration by design |
| `sum_of_stages_us` | the serial pass: every node run alone and synchronised, summed |
| `measured_gpu_execution_us` | GPU execution summed over the nodes, from the GPU's own timestamps (Metal command buffers, or CUPTI on CUDA) |
| `cross_stream_stall_us` | time streams waited on each other's data, summed (can exceed the wall when streams stall at once) |
| `executed_node_count`, `gpu_measured_node_count`, `stream_count` | counts |

72 bytes. Mirrors in other languages must match it field for field.

### mlx_perf_variant

```c
typedef struct mlx_perf_variant {
  const char* variant_name;
  mlx_perf_build_graph_fn build_graph;
  void* builder_payload;
} mlx_perf_variant;
```

One implementation in a comparison. `variant_name` becomes part of file names: non-empty, no `/`, and different from the other variant's ignoring case.

### mlx_perf_metric_comparison

```c
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
  double median_difference_us;
  double difference_interval_low_us;
  double difference_interval_high_us;
  double difference_interval_coverage;
  double sign_test_p_value;
  double relative_difference;
  int verdict;
  int rounds_candidate_faster;
  int round_count;
} mlx_perf_metric_comparison;
```

One metric across the paired rounds, in microseconds. `median_difference_us` is the median per-round candidate − baseline: negative means the candidate took less time. `difference_interval_low_us` to `difference_interval_high_us` is the distribution-free interval for that median, with `difference_interval_coverage` (0.961 for 12 rounds). `sign_test_p_value` is the exact two-sided sign test over the non-zero per-round differences. `relative_difference` is `median_difference_us` over the baseline median. `verdict` is one of the `MLX_PERF_VERDICT_*` constants. `rounds_candidate_faster` of `round_count` counts the rounds the candidate won. 144 bytes.

### mlx_perf_comparison_summary

```c
typedef struct mlx_perf_comparison_summary {
  mlx_perf_job_summary baseline;
  mlx_perf_job_summary candidate;
  mlx_perf_metric_comparison job_wall;
  mlx_perf_metric_comparison gpu_span;
  mlx_perf_metric_comparison host_submission;
  double max_absolute_output_difference;
} mlx_perf_comparison_summary;
```

`job_wall` is the job's uninstrumented duration (the headline), `gpu_span` the GPU time from the job's first command buffer's start to its last one's end summed over streams, `host_submission` the host time inside the job's `async_eval`. `max_absolute_output_difference` is the largest element-wise difference between the two variants' outputs, computed in float32, or `NaN` when any difference is not finite; it is reported, not judged. 584 bytes.

### mlx_perf_procedure_step_fn

```c
typedef int (*mlx_perf_procedure_step_fn)(void* procedure_payload);
```

A procedure's `prepare` or `run` callback. Return `0` on success; any other value stops the comparison with `procedure <name>: run returned <n>` (or `prepare returned <n>`).

### mlx_perf_procedure

```c
typedef struct mlx_perf_procedure {
  const char* procedure_name;
  mlx_perf_procedure_step_fn prepare;
  mlx_perf_procedure_step_fn run;
  void* procedure_payload;
} mlx_perf_procedure;
```

One procedure in a procedure comparison. `run` is measured from its start until every command buffer it committed has completed. `prepare` (may be `NULL`) runs before every run, untimed; evaluate what it creates before returning, so that work is not left to the timed run. Both are called on the calling thread with `procedure_payload`; `run` may hand its MLX work to other threads, since everything the process submits to the GPU while it runs is counted as its own. `procedure_name` follows the rules of `variant_name`. 32 bytes.

### mlx_perf_procedure_summary

```c
typedef struct mlx_perf_procedure_summary {
  double median_command_buffer_count;
  double median_blocking_wait_count;
  double median_blocking_wait_us;
  int kernel_launch_count;
  int launches_unable_to_fill_gpu;
} mlx_perf_procedure_summary;
```

One procedure's hand-offs: the medians over the 12 rounds of its command buffers, its blocking waits (evaluations and synchronizes that stopped the host until the GPU finished) and the host time they took. `kernel_launch_count` is from the observed run; `launches_unable_to_fill_gpu` counts its launches with fewer threadgroups than GPU cores, or is `-1` when the core count is unknown. 32 bytes.

### mlx_perf_procedure_comparison_summary

```c
typedef struct mlx_perf_procedure_comparison_summary {
  mlx_perf_procedure_summary baseline;
  mlx_perf_procedure_summary candidate;
  mlx_perf_metric_comparison duration;
  mlx_perf_metric_comparison gpu_busy;
  mlx_perf_metric_comparison gpu_idle;
  mlx_perf_memory_comparison peak_mlx_active_added;
  mlx_perf_memory_comparison mlx_held_change;
  mlx_perf_memory_comparison system_wired_at_end;
  int baseline_rounds_crossing_wire_limit;
  int candidate_rounds_crossing_wire_limit;
  int baseline_rounds_with_compression;
  int candidate_rounds_with_compression;
} mlx_perf_procedure_comparison_summary;
```

`duration` is the run until its GPU work completed (the headline), `gpu_busy` the time inside it when at least one of its command buffers was on the GPU, `gpu_idle` the rest: the GPU waiting on the host. The three memory comparisons are judged on the same rounds, from figures read just before and just after each run. `peak_mlx_active_added` is MLX's peak active memory during the run less its active memory at the start. `mlx_held_change` is MLX's active memory plus cache at the end less at the start. `system_wired_at_end` is the system's wired total as the run ended. The four counts say in how many of the 12 rounds each side's run crossed the wire limit (the kernel refused wire requests over it, or the wired total ended over it) and saw the system compress memory (any program's). 704 bytes; 0.5.0's was 496.

### mlx_perf_memory_comparison

```c
typedef struct mlx_perf_memory_comparison {
  double baseline_median_bytes;
  double candidate_median_bytes;
  double median_difference_bytes;
  double difference_interval_low_bytes;
  double difference_interval_high_bytes;
  double relative_difference;
  int verdict;
  int rounds_candidate_less;
  int round_count;
} mlx_perf_memory_comparison;
```

One memory figure across the paired rounds, in bytes, judged like time: lower is better. `median_difference_bytes` is the median per-round candidate − baseline, negative when the candidate used less; the interval and `verdict` work exactly as in `mlx_perf_metric_comparison`, where `MLX_PERF_VERDICT_CANDIDATE_FASTER` means the candidate used less. 64 bytes.

### mlx_perf_wired_memory

```c
typedef struct mlx_perf_wired_memory {
  uint64_t system_wired_bytes;
  uint64_t wire_limit_bytes;
  uint64_t mlx_active_bytes;
  uint64_t mlx_cache_bytes;
  int64_t wired_outside_mlx_bytes;
  int64_t wire_headroom_bytes;
} mlx_perf_wired_memory;
```

One reading. `system_wired_bytes` is `vm.page_wired_count` times the page size; `wire_limit_bytes` is `vm.global_user_wire_limit`, or 0 when unknown. `wired_outside_mlx_bytes` is the wired total less MLX's active memory and cache (it may be negative); `wire_headroom_bytes` is the limit less the wired total, or 0 when the limit is unknown. 48 bytes.

### mlx_perf_wire_release_lag

```c
typedef struct mlx_perf_wire_release_lag {
  uint64_t allocation_bytes;
  uint64_t wire_limit_bytes;
  int64_t median_wired_rise_bytes;
  double median_release_lag_us;
  double largest_release_lag_us;
  int repetition_count;
  int resolved_count;
} mlx_perf_wire_release_lag;
```

The probe's result. `allocation_bytes` is the size measured, rounded down to whole float32 elements. `median_wired_rise_bytes` is how far the wired total rose while the allocation was held. The lags are over the `resolved_count` repetitions whose un-wiring was seen in time, `NaN` when none was. 48 bytes.

## Constants

### Verdicts

```c
enum {
  MLX_PERF_VERDICT_CANDIDATE_FASTER = -1,
  MLX_PERF_VERDICT_WITHIN_SPREAD = 0,
  MLX_PERF_VERDICT_CANDIDATE_SLOWER = 1
};
```

The value of `mlx_perf_metric_comparison.verdict`. Lower time is better. `MLX_PERF_VERDICT_CANDIDATE_FASTER` only when the whole interval for the median per-round difference lies below zero (the candidate won at least 10 of 12 rounds); `MLX_PERF_VERDICT_CANDIDATE_SLOWER` when it lies above; otherwise `MLX_PERF_VERDICT_WITHIN_SPREAD`, which means no result. Two identical variants are called different less than 4% of the time per metric.

### Version macros

```c
#define MLX_PERF_VERSION_MAJOR 0
#define MLX_PERF_VERSION_MINOR 6
#define MLX_PERF_VERSION_PATCH 0
#define MLX_PERF_VERSION_STRING "0.7.0"
```

From `mlx_perf/mlx_perf_version.h`, generated from the project version, so the header, the library and the installed package agree. Before 1.0, a minor version may change the C API; the release notes say how.
