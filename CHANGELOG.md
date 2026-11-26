# Changelog

Versions follow [`VERSIONING.md`](VERSIONING.md). Each release lists the
MLX releases it was built and tested against.

## Unreleased

Additive: the next release is a minor one (0.2.0) under `VERSIONING.md`.

- Per-node measurements as JSON beside every report (schema version 1,
  [`docs/measurement-file.md`](docs/measurement-file.md)), and the new C
  function `mlx_perf_profile_to_measurements` that writes only that file
  (#11).
- The file records each job-wall sample, the time spent inside the job's
  graph-building code per pass, and the command buffers the timeline pass
  observed.
- Custom Metal kernels are told apart by name (`kernel_name`).
- Comparing two implementations of one job (#12): the C function
  `mlx_perf_compare_to_report` (and `measure_pair` in C++) checks the two
  outputs agree, interleaves the passes with the order flipped every round,
  and judges job duration, GPU execution and host submission against the
  spread (the middle halves of the rounds must not overlap). The comparison
  page opens with the verdict; `<base>.json` records it.
- `hello_world` and `qwen35_profile` measure eager vs compiled this way.

## 0.1.0 (2026-11-26)

Tested MLX releases: 0.32.4, 0.32.5 (Metal, macOS).

First versioned release of the library:

- C API `mlx_perf_profile_to_report`, `mlx_perf_last_error` and
  `mlx_perf_version`, in `include/mlx_perf/mlx_perf_c_api.h`.
- An installable `libmlx_perf.a` with a `find_package(mlx_perf)` package
  that links MLX from the consumer's own `MLX_PREFIX`.
- The Spark-style HTML report: job summary, per-stream timeline, task
  table and DAG, per node or per job-supplied stage.
