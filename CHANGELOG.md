# Changelog

Versions follow [`VERSIONING.md`](VERSIONING.md). Each release lists the
MLX releases it was built and tested against.

## 0.1.0 (2026-11-26)

Tested MLX releases: 0.32.4, 0.32.5 (Metal, macOS).

First versioned release of the library:

- C API `mlx_perf_profile_to_report`, `mlx_perf_last_error` and
  `mlx_perf_version`, in `include/mlx_perf/mlx_perf_c_api.h`.
- An installable `libmlx_perf.a` with a `find_package(mlx_perf)` package
  that links MLX from the consumer's own `MLX_PREFIX`.
- The Spark-style HTML report: job summary, per-stream timeline, task
  table and DAG, per node or per job-supplied stage.
