# Getting started

From a fresh checkout to an HTML report of an MLX job, in four steps.

## What you need

- A Mac with Apple silicon (MLX's Metal backend), CMake 3.24 or newer, and a C++20 compiler (Xcode's command line tools).
- A **prebuilt MLX prefix** of MLX 0.32.4 or 0.32.5: one directory holding MLX and MLX-C together, that is `lib/libmlx.a`, `lib/libmlxc.a`, `include/mlx/`, `include/mlx/c/` and `include/metal_cpp/`.
- Python 3 only if you want to run the example that reads the JSON.

## Step 1: get an MLX prefix

The simplest source is [mlx-bridge](https://github.com/aosama/mlx-bridge), whose first build compiles MLX and MLX-C into a shared prefix under `~/.cache/mlx-bridge/native/<identity>/prefix`. Any `cargo build` of a crate that depends on it creates one; its [native build guide](https://github.com/aosama/mlx-bridge/blob/main/docs/native-build.md) explains the identity and the options.

Alternatively, build [MLX](https://github.com/ml-explore/mlx) and [MLX-C](https://github.com/ml-explore/mlx-c) as static libraries yourself and install both into the same prefix with `cmake --install`.

## Step 2: build the profiler

```sh
git clone --branch v0.4.0 https://github.com/aosama/mlx-performance-profiler.git
cd mlx-performance-profiler
cmake -S . -B build -DMLX_PREFIX=/path/to/mlx/prefix
cmake --build build
```

Configure prints the MLX release it found and warns, without failing, when it is not a tested one. `MLX_PREFIX` must be given on the command line: it is a CMake cache variable and is never read from the environment.

## Step 3: profile the bundled demo

```sh
./build/hello_world
open build/hello_world.html
```

`hello_world` measures a small three-stream job two ways, eager and compiled with `mlx::core::compile`, and writes:

| File | What it is |
|---|---|
| `build/hello_world.html` | the comparison, opening with the verdict: which variant is faster and whether that beats the noise |
| `build/hello_world_eager.html`, `build/hello_world_compiled.html` | the full report for each variant: summary, per-stream timeline, task table, interactive execution graph |
| `build/hello_world_eager.json`, `build/hello_world_compiled.json` | the same measurements as data ([format](measurement-file.md)) |
| `build/hello_world.json` | the comparison as data |

## Step 4: profile your own job

Your job is a C callback that builds an MLX graph with MLX-C. This program profiles `softmax(x @ x)`. It is [`examples/profile_one_job.c`](examples/profile_one_job.c), and the test suite builds and runs it:

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
  /* Before any MLX work, so the report names every kernel launch it records. */
  mlx_perf_install_launch_recorder();
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

Build it as part of your own CMake project with [`examples/CMakeLists.txt`](examples/CMakeLists.txt), which fetches this library at the release tag (see [Building and linking](building-and-linking.md)). The examples folder is in the clone from step 2, at the same tag as the library it fetches. From that clone's root:

```sh
cd docs/examples
cmake -S . -B build -DMLX_PREFIX=/path/to/mlx/prefix
cmake --build build
./build/profile_one_job build
open build/softmax_matmul.html
```

It prints a line like `job duration 733.9 us (median), GPU execution 586.8 us, 2 nodes on 1 streams` and writes `softmax_matmul.html` and `softmax_matmul.json`.

## Next

- [Profiling a job](profiling-a-job.md): the callback contract, streams, and what each pass measures.
- [Comparing two implementations](comparing-implementations.md): is your version faster?
- [Reading the report](reading-the-report.md): what the timeline and the graph are telling you.
