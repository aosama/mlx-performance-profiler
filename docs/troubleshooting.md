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

`MLX_PREFIX` must hold MLX and MLX-C together: `include/mlx/`, `include/mlx/c/`, `include/metal_cpp/`, `lib/libmlx.a` and `lib/libmlxc.a`. See [Getting started](getting-started.md#step-1-get-an-mlx-prefix).

### An untested MLX release

> configure: `is not a tested release (tested: <releases>). It may still work`

A warning only: a source build continues. The tested releases are 0.32.4 and 0.32.5. The README's MLX compatibility section lists the five MLX internals the library relies on.

### The installed package was built against another MLX

> configure: `this mlx_perf was built against MLX <release>, but MLX_PREFIX holds MLX <release>.`

An installed `libmlx_perf.a` is compiled against one exact MLX release. Point `MLX_PREFIX` at that release, or build the library from source with FetchContent against yours (see [Building and linking](building-and-linking.md)).

### The package refuses the version you asked for

`find_package(mlx_perf 0.2)` does not accept 0.3.x: before 1.0, each minor release may change the C API. Ask for the minor version you installed, for example `find_package(mlx_perf 0.3 REQUIRED)`.

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

### A field is `null` in the JSON

`null` means not measured, never zero. A node on a CPU stream has no GPU window; a node in a stage-timed run is timed as part of its stage; a serial time is `null` when the serial pass ran a different operation at that position. See [The measurement file](measurement-file.md#conventions).

### The report's graph says it needs JavaScript

The execution graph is drawn by JavaScript embedded in the report. Open the file in a browser with JavaScript enabled; the rest of the report reads without it.
