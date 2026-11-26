# Profiling a job

How to measure your own MLX computation and get its report: the callback you write, what the profiler does with it, and what comes back.

## Profile an MLX graph from C

Call `mlx_perf_profile_to_report` with a callback that builds your graph. This complete program profiles `softmax(x @ x)` on a 1024 × 1024 input and writes `softmax_matmul.html` and `softmax_matmul.json`:

<!-- example: docs/examples/profile_one_job.c -->
```c
/*
 * Profile one MLX job from C and write its HTML report and measurement file.
 *
 * The job: scores = softmax(x @ x) over the last axis, for a 1024 x 1024
 * input. Build:  see docs/examples/CMakeLists.txt.
 * Run:           ./profile_one_job <output directory>
 * Writes:        <output directory>/softmax_matmul.html and .json
 */
#include "mlx/c/array.h"
#include "mlx/c/ops.h"
#include "mlx/c/random.h"
#include "mlx/c/stream.h"
#include "mlx_perf/mlx_perf_c_api.h"

#include <stdio.h>

typedef struct softmax_job {
  mlx_array input;   /* materialized once, before profiling */
  mlx_stream stream; /* where the job's operations run */
} softmax_job;

/* Called many times by the profiler (11 for one profile). Each call must
 * append the outputs of a FRESH, unevaluated graph: the profiler evaluates
 * them itself, once per pass. Inputs may be reused; operations may not. */
static int build_softmax_job(mlx_vector_array fresh_outputs, void* builder_payload) {
  const softmax_job* job = (const softmax_job*)builder_payload;
  mlx_array products = mlx_array_new();
  mlx_array scores = mlx_array_new();
  int status = mlx_matmul(&products, job->input, job->input, job->stream);
  status |= mlx_softmax_axis(&scores, products, -1, false, job->stream);
  status |= mlx_vector_array_append_value(fresh_outputs, scores);
  mlx_array_free(products);
  mlx_array_free(scores);
  return status; /* non-zero makes the profiler stop and report the failure */
}

int main(int argument_count, char** arguments) {
  const char* output_directory = argument_count > 1 ? arguments[1] : ".";
  const int input_shape[] = {1024, 1024};
  softmax_job job = {mlx_array_new(), mlx_default_gpu_stream_new()};
  mlx_array random_key = mlx_array_new();
  mlx_random_key(&random_key, 7);
  mlx_random_normal(&job.input, input_shape, 2, MLX_FLOAT32, 0.0f, 1.0f, random_key, job.stream);
  mlx_array_eval(job.input); /* inputs hold data; only the job's operations are timed */

  mlx_perf_job_summary summary;
  const int status = mlx_perf_profile_to_report(build_softmax_job, &job, "eager",
                                                "softmax(x @ x), 1024 x 1024",
                                                output_directory, "softmax_matmul", &summary);
  if (status == 0) {
    printf("job duration %.1f us (median), GPU execution %.1f us, %d nodes on %d streams\n",
           summary.job_wall_median_us, summary.measured_gpu_execution_us,
           summary.executed_node_count, summary.stream_count);
  } else {
    fprintf(stderr, "profiling failed: %s\n", mlx_perf_last_error());
  }
  mlx_array_free(random_key);
  mlx_array_free(job.input);
  mlx_stream_free(job.stream);
  return status == 0 ? 0 : 1;
}
```

From C++, include the same header: it is `extern "C"`, and MLX-C's `mlx_array` wraps the `mlx::core::array` your C++ code already builds.

## The callback contract

```c
typedef int (*mlx_perf_build_graph_fn)(mlx_vector_array fresh_outputs, void* builder_payload);
```

- **Build a fresh, unevaluated graph on every call** and append its outputs to `fresh_outputs`. The profiler calls the callback once per graph it needs: 11 times for one profile (stream discovery, warm-up, 7 timed runs, the timeline pass and the serial pass), and more per variant in a comparison. It evaluates each graph itself. An output that is already evaluated has nothing left to measure.
- **Reuse inputs, not operations.** Create and `mlx_array_eval` your inputs (weights, activations) once, before profiling, and pass them in `builder_payload`. Inside the callback, create only the operations. Evaluated inputs are the graph's leaves and are not timed; generating them inside the callback would time their creation as part of the job.
- **Return 0 on success.** Any other value stops the profile, and `mlx_perf_last_error()` then reads `build_graph callback returned <your value>`. An empty `fresh_outputs` fails with `build_graph callback produced no outputs`.
- **Own nothing across the call.** The library creates and frees `fresh_outputs`; MLX-C copies a handle on append, so free your own handles before returning, as the example does.
- **Stay synchronous and on one thread.** The profiler calls the callback on the thread that called it, and `builder_payload` must stay alive for the whole call.

## Streams

Operations run on the stream you pass to each MLX-C call. To see work overlap, and stall on each other, put branches of the job on different streams:

```c
mlx_stream main_stream = mlx_default_gpu_stream_new();
mlx_stream side_stream = mlx_stream_new_device(mlx_device_new_type(MLX_GPU, 0));
/* mlx_matmul(&product, a, b, side_stream); mlx_exp(&activated, product, main_stream); */
```

The profiler discovers the streams from the graph itself; you never register them. A node on one stream that reads another stream's output is a cross-stream join, and the time it waits for that data is reported as **stalled on another stream**.

## What the profiler does with your job

| Pass | What it measures |
|---|---|
| Stream discovery | builds one graph to find the streams; not timed |
| Warm-up | runs the job once; discarded, it absorbs Metal kernel compilation |
| Job duration | 7 uninstrumented runs, each one `async_eval` with streams overlapping: the median and range are the job's real duration |
| Timeline | one `async_eval` per node, with the GPU's own start and end read from each node's Metal command buffer: the six slices of every node, and the per-stream timeline |
| Serial | each node run alone and synchronised: its isolated cost and memory |

The timeline pass submits nodes one at a time so it can time each one, which adds submission cost a plain `eval` batches away. That is why the report shows both the instrumented duration and the uninstrumented job duration, and never presents one as the whole truth.

## What comes back

- `<output_directory>/<report_file_base>.html`: the report ([Reading the report](reading-the-report.md)).
- `<output_directory>/<report_file_base>.json`: the same numbers as data ([The measurement file](measurement-file.md)).
- `<output_directory>/<report_file_base>.dot`: the plan as a Graphviz digraph, for anyone who wants to render it themselves.
- The `mlx_perf_job_summary` you pass (or `NULL`): headline numbers in microseconds.

| Summary field | Meaning |
|---|---|
| `job_wall_median_us`, `job_wall_min_us`, `job_wall_max_us` | the uninstrumented job duration and its range |
| `timeline_wall_us` | the instrumented timeline pass, first issue to last completion |
| `sum_of_stages_us` | the serial pass summed: every node run alone |
| `measured_gpu_execution_us` | GPU execution summed over the nodes |
| `cross_stream_stall_us` | time streams spent waiting on each other's data, summed |
| `executed_node_count`, `gpu_measured_node_count`, `stream_count` | counts |

The output directory must exist; the library does not create it, and a write failure returns non-zero with a message naming the directory.

## Large graphs and stages

A graph of about a thousand nodes, such as a decoder step, is better timed one stage (for example one layer) at a time, because one command buffer per node would cost more than the job itself. The C API times every node: stage timing is part of this repository's internal C++ code and is not yet in the installed API. `qwen35_profile`, built with this repository's tools, shows it on Qwen3.5. See [Reading the report](reading-the-report.md#stage-timed-reports) for how a stage-timed report reads.

## Related

- [Comparing two implementations](comparing-implementations.md) measures two callbacks against each other.
- [Measuring from code](measuring-from-code.md) writes only the JSON, with console output off.
- [C API reference](reference/c-api.md#mlx_perf_profile_to_report) for exact signatures.
