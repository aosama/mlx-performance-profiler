# Troubleshooting

Find the message you got, or the number that surprised you. Every quoted message below is checked by the test suite against the library's source, so it is the exact text you will see (`<...>` marks the part that varies).

## Errors from the C API

After a call returns non-zero, `mlx_perf_last_error()` holds one of these messages.

### The callback reported a failure

> error: `build_graph callback returned <value>`

Your `build_graph` callback returned non-zero. Usually an MLX-C call inside it failed: MLX-C functions return non-zero on failure, and the examples OR their statuses together. Check the shapes and dtypes your operations receive, and that every input array is valid for the whole call.

### The callback appended nothing

> error: `build_graph callback produced no outputs`

The callback returned 0 but appended no array to `fresh_outputs`. Append the job's output with `mlx_vector_array_append_value(fresh_outputs, output)`.

### A required argument is missing

> error: `<entry point>: null argument`

One of the pointer arguments, for example `variant_name`, `output_directory` or the callback, is `NULL`. Only `summary` may be `NULL`.

### The output directory cannot be written

> error: `cannot write the report into <directory>`

> error: `cannot write the measurements into <directory>`

> error: `cannot write the comparison into <directory>`

The directory does not exist, is not writable, or the disk is full. The library does not create directories: create it before calling.

### A comparison variant is missing

> error: `mlx_perf_compare_to_report: null variant`

`baseline` or `candidate` is `NULL`.

### A variant name cannot be used as a file name

> error: `mlx_perf_compare_to_report: variant name "<name>" must be non-empty without '/'`

Variant names become part of file names (`<base>_<name>.html`). Use a short name such as `stock` or `custom_kernel`.

### Both variants are named the same

> error: `mlx_perf_compare_to_report: both variants are named "<name>" (ignoring case)`

The two names differ only in case, or not at all. Their files would overwrite each other on macOS's case-insensitive file system. Give them distinct names.

### The two implementations do not produce the same outputs

> error: `output <index> differs: baseline <shape and dtype>, candidate <shape and dtype>`

> error: `the baseline produces <count> outputs, the candidate <count>`

A comparison measures two ways of computing the same thing, so both must append the same number of outputs, in the same order, with the same shapes and dtypes. Nothing is measured or written. Values may differ: the largest element-wise difference is reported, not judged.

### A procedure step reported a failure

> error: `procedure <name>: run returned <value>`

> error: `procedure <name>: prepare returned <value>`

Your procedure's `run` or `prepare` callback returned non-zero, so the comparison stopped and wrote nothing. As with a graph callback, an MLX-C call inside it usually failed; the message names the procedure and the step.

### A procedure or its run callback is missing

> error: `mlx_perf_compare_procedures_to_report: null procedure`

> error: `mlx_perf_compare_procedures_to_report: null run callback`

`baseline` or `candidate` is `NULL`, or its `run` is. Only `prepare` and `summary` may be `NULL`.

### A procedure name cannot be used

> error: `procedure name "<name>" must be non-empty without '/'`

> error: `both procedures are named "<name>" (ignoring case)`

Procedure names follow the rules of variant names: non-empty, no `/`, and distinct from each other ignoring case.

### Procedures cannot be measured on this GPU

> error: `cannot observe this GPU's submissions: procedures cannot be measured`

The profiler could not observe how MLX submits work to the GPU and waits for it. On macOS it could not find the Metal command buffer and shared event classes MLX uses, for example on a Mac without a Metal GPU. On Linux, CUPTI refused the subscription: there is no CUDA device, or another profiler, such as Nsight Systems or Nsight Compute, already holds this process's CUPTI subscription (only one is allowed). Without these observations every run would read as all GPU idle, so the comparison refuses instead of reporting that. Graph comparisons (`mlx_perf_compare_to_report`) do not need them.

### The procedure comparison cannot be written

> error: `cannot write the procedure comparison into <directory>`

As for the other reports: the directory does not exist, is not writable, or the disk is full.

### The system's wired memory cannot be read

> error: `cannot read the system's wired memory`

`mlx_perf_read_wired_memory` got no answer from the kernel for the wired total: both `sysctl vm.page_wired_count` and `host_statistics64` were refused, for example by a sandbox. In the same situation a report's wired memory lane shows a wired total of 0.

### The wire release probe refused its request

> error: `wire release probe: allocation_bytes and repetition_count must be positive`

> error: `wire release probe: allocation_bytes must be under 8 GiB`

`mlx_perf_measure_wire_release_lag` needs at least one float32 element (4 bytes) and at least one repetition. One allocation must also fit one MLX array, whose shape holds at most 2³¹ − 1 elements: under 8 GiB of float32.

### An MLX-C handle could not be read

> error: `mlx_vector_array handle is null`

> error: `cannot read element <index> of an mlx_vector_array`

The vector or an array appended to it is not a valid MLX-C handle, for example an array already freed, or one never assigned by an MLX-C call. Append arrays that an MLX-C operation produced, before freeing them.

## Configure and link problems

### No MLX prefix was given

> configure: `MLX_PREFIX is required but was not set.`

Add `-DMLX_PREFIX=/path/to/prefix` to the `cmake` command. It is never read from an environment variable.

### The prefix holds no MLX

> configure: `does not contain include/mlx/mlx.h. Point it at a prebuilt MLX prefix (containing include/ and lib/).`

`MLX_PREFIX` must hold MLX and MLX-C together: `include/mlx/`, `include/mlx/c/`, `include/metal_cpp/`, `lib/libmlx.a` and `lib/libmlxc.a`. See [Getting started](getting-started.md#step-1-get-an-mlx-prefix). On Linux the prefix holds MLX's CUDA build instead; see [Building and linking](building-and-linking.md#on-linux-with-an-nvidia-gpu).

### The prefix's MLX has no GPU backend to measure

> configure: `holds an MLX built without a GPU backend this library measures.`

Off macOS the library measures MLX's CUDA backend only. Build MLX with `-DMLX_BUILD_CUDA=ON` and install it into the prefix; its `share/cmake/MLX/MLXConfig.cmake` then finds `CUDAToolkit`, which is how configure recognises it. An MLX built for its CPU backend alone has no GPU to measure.

### The CUDA prefix has no fmt headers

> configure: `has no include/fmt/format.h.`

MLX's CUDA headers include `<fmt/format.h>`, and MLX's install leaves fmt out. Copy the fmt headers MLX was built with, from its build tree's `_deps/fmt-src/include/fmt`, into `<prefix>/include/fmt`.

### An untested MLX release

> configure: `is not a tested release (tested: <releases>). It may still work`

A warning only: a source build continues. The tested releases are 0.32.4 and 0.32.5. The README's MLX compatibility section lists the five MLX internals the library relies on.

### The installed package was built against another MLX

> configure: `this mlx_perf was built against MLX <release>, but MLX_PREFIX holds MLX <release>.`

An installed `libmlx_perf.a` is compiled against one exact MLX release. Point `MLX_PREFIX` at that release, or build the library from source with FetchContent against yours (see [Building and linking](building-and-linking.md)).

### The package refuses the version you asked for

`find_package(mlx_perf 0.6)` does not accept 0.7.x: before 1.0, each minor release may change the C API. Ask for the minor version you installed, for example `find_package(mlx_perf 0.7 REQUIRED)`.

### Undefined C++ symbols when linking a C program

Enable C++ in your project, `project(my_app LANGUAGES C CXX)`, so CMake links with the C++ driver: the library is C++ inside.

### Duplicate or mismatched MLX symbols at link or run time

Two different MLX builds are in one binary, for example one from your prefix and one from another dependency. Build everything against the same `MLX_PREFIX`.

## Surprising numbers

### The instrumented duration is much longer than the job duration

Expected. The timeline pass submits each node on its own so it can time it, which adds per-node submission cost that a plain `eval` batches away. Quote the job duration. On graphs of hundreds of nodes, stage timing keeps this cost small (see [Reading the report](reading-the-report.md#stage-timed-reports)).

### The verdict says "within the spread"

The difference between the two implementations did not exceed run-to-run noise: no result either way. Measure on a quiet machine (nothing else using the GPU), or measure a larger job.

### Two runs of the same job give different durations

Expected: clocks, temperature and other work on the machine move every measurement. Never compare durations from separate runs to decide what is faster; use [`mlx_perf_compare_to_report`](comparing-implementations.md).

### A node shows 0.0 µs compute

Operations that only describe a view of existing memory, such as a broadcast or a squeeze, run no kernel and show 0.0 µs compute. Their wall time is waiting, which their classification shows.

### A kernel launch has no `kernel_function_name`

The pipeline's name is captured when MLX compiles it. A kernel compiled before the profiler started observing (by MLX work your program did before its first profile) is recorded with its shape but `null` for its name. Call `mlx_perf_install_launch_recorder()` before your program's first MLX evaluation to name every launch. See the [C API reference](reference/c-api.md#mlx_perf_install_launch_recorder).

### Many launches are flagged ⚠ in a decode step

Expected for small inputs: an element-wise operation on one token's vector forms a handful of threadgroups, so it cannot use every GPU core. On a Qwen3.5 decode step about 7 in 10 launches are like this. Look for the expensive ones: the hot path, cost colours, or a large kernel such as a matrix multiply that forms only a few threadgroups.

### "Wired outside MLX" jumps after a release, or drops while nothing happens

Both are macOS, not the profiler. Memory MLX releases stays wired for a fraction of a second, so "outside MLX" rises by the released amount until macOS un-wires it; the wired memory lane marks those samples. Memory MLX still holds is un-wired after about a second without GPU use, so "outside MLX" falls, and it rises again when the GPU next uses that memory. See [Wired memory](wired-memory.md).

### The memory verdict's working set is smaller than the arrays the run made

The peak MLX working set is MLX's peak active memory during the run less its active memory as the run began. Memory already active at the start, such as weights or the other side's arrays, is not part of it. The raw peak is in each run's `memory.peak_mlx_active_bytes`.

### A note says rounds crossed the wire limit or compressed memory

In those rounds the kernel refused wire requests over `vm.global_user_wire_limit`, the wired total ended over it, or the system compressed memory. Such rounds are slow for a reason outside the code, so a timing verdict that rests on them says more about the machine than the change. Compression is counted system-wide: close other memory-heavy programs and compare again. Each round's figures are in the JSON under `memory`.

### A release lag is `null`

macOS had not un-wired the released memory before sampling stopped (3 s after a procedure's observed run; the end of the serial pass for a report). The lag is longer than the profiler waited, or something else wired memory at the same moment. Run `mlx_perf_measure_wire_release_lag` on a quiet machine to measure it alone.

### A field is `null` in the JSON

`null` means not measured, never zero. A node on a CPU stream has no GPU window; a node in a stage-timed run is timed as part of its stage; a serial time is `null` when the serial pass ran a different operation at that position. See [The measurement file](measurement-file.md#conventions).

### The report's graph says it needs JavaScript

The execution graph is drawn by JavaScript embedded in the report. Open the file in a browser with JavaScript enabled; the rest of the report reads without it.
