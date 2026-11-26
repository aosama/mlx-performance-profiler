# mlx-performance-profiler documentation

mlx-performance-profiler measures what Apple [MLX](https://github.com/ml-explore/mlx) actually did when it evaluated your computation: every operation, which GPU stream it ran on, and how each microsecond split between **computing** and **waiting** (on the host, on earlier work on the same stream, or on another stream's data). It writes a self-contained HTML report with an interactive execution graph, and the same numbers as JSON. Given two implementations of one job, it says which is faster and whether the difference beats run-to-run noise.

It is a C++ library with a C API, used from C, C++ and Rust (through [mlx-bridge](https://github.com/aosama/mlx-bridge)), on macOS with Apple silicon and MLX's Metal backend. It observes MLX; it never changes what MLX computes.

Every C program and CMake project on these pages is a file in [`examples/`](examples/) that the test suite builds and runs, and every measurement file it writes is validated against the JSON Schemas in [`schema/`](schema/). A test also fails if a page's copy of an example drifts from the file.

## Which page answers your question

| You want to… | Read |
|---|---|
| Build it and see a report in five minutes | [Getting started](getting-started.md) |
| Profile your own MLX graph from C or C++ | [Profiling a job](profiling-a-job.md) |
| Know whether your kernel or rewrite is really faster | [Comparing two implementations](comparing-implementations.md) |
| Compare two loops or pipelines that evaluate many graphs (a token loop, a sampler) | [Comparing procedures](comparing-procedures.md) |
| Know how much room the GPU has under macOS's wire limit, and when "wired outside MLX" misleads | [Wired memory](wired-memory.md) |
| Measure inside a test, benchmark or CI job, silently | [Measuring from code](measuring-from-code.md) |
| Read per-node numbers as data (Python, jq, any language) | [The measurement file](measurement-file.md) |
| Understand the report: the six slices, the timeline, the graph | [Reading the report](reading-the-report.md) |
| Add it to your CMake project, or install it | [Building and linking](building-and-linking.md) |
| Use it from Rust | [Using it from Rust](using-from-rust.md) |
| Fix an error message or a surprising number | [Troubleshooting](troubleshooting.md) |
| Look up an exact signature or struct field | [C API reference](reference/c-api.md) |

## Five facts that prevent most mistakes

1. **Your callback builds a fresh graph every time it is called.** The profiler calls it many times (11 for one profile) and evaluates each graph itself. Build inputs once, outside; build the operations inside. See [Profiling a job](profiling-a-job.md#the-callback-contract).
2. **Never decide "faster" from two separate runs.** Use `mlx_perf_compare_to_report` (one graph each) or `mlx_perf_compare_procedures_to_report` (any code that evaluates), which interleave the two and judge the per-round differences. See [Comparing two implementations](comparing-implementations.md) and [Comparing procedures](comparing-procedures.md).
3. **"Within the spread" is not a result.** It means the difference did not beat noise. Report it as no difference found.
4. **`null` in the JSON means not measured, never zero.** See [The measurement file](measurement-file.md#conventions).
5. **`MLX_PREFIX` is a CMake cache variable**, the prefix of a prebuilt MLX (0.32.4 or 0.32.5), and your binary must link exactly that one MLX. See [Building and linking](building-and-linking.md).

## Versions

These pages describe the current release, 0.5.0 (MLX 0.32.4 and 0.32.5). 0.4.0 and 0.5.0 added to the C API (procedure comparisons, `mlx_perf_install_launch_recorder`, and the wired memory reading and probe) without changing any existing function or struct; the measurement file is still `schema_version` 1, with new fields only. What changed in each release is in the [changelog](../CHANGELOG.md) and the [release notes](https://github.com/aosama/mlx-performance-profiler/releases).
