# Instructions for maintaining mlx-performance-profiler

## What this repo is

An experiment: reproduce the useful parts of Apache Spark's Web UI
("what did the engine actually do") for Apple MLX computations, by observing
MLX's lazy computational graph and adding the telemetry Spark shows per stage.

The engine API surface lives in the sibling project `../mlx-bridge`
(the `mlx-bridge` Rust crate over MLX-C). This repo is the profiling
sidecar: it observes, it never changes MLX semantics.

## Conventions

- Naming: the repository is `mlx-performance-profiler`; code uses the short
  prefix `mlx_perf` (C functions `mlx_perf_*`, the C++ namespace and
  `libmlx_perf.a`) so call sites stay readable.
- C++20, one job per file: graph capture, telemetry, DOT writing, SVG
  rendering, and HTML report writing are separate modules with separate
  headers. `main`-level files only orchestrate.
- A single function stays around 40 lines; split at natural seams when longer.
- Comments carry rationale (why this design, what tradeoff), not narration.
- Variable names describe domain meaning: `output_node_ids`,
  `stream_index_to_stream`, `node_id_to_telemetry`. No `tmp`/`data`/`info`.
- Any run longer than a few seconds prints live progress with an ETA.
- The prebuilt MLX the spike links against lives at
  `~/.cache/mlx-bridge/native/<identity>/prefix` and is passed as the CMake
  cache variable `MLX_PREFIX` — never as an environment variable.
- Targets: `mlx_perf` (profiler, reports, C API) is the only installed
  library and never depends on the demo, server or model code, which build
  as `mlx_perf_demo_model`, `mlx_perf_live_view` and `mlx_perf_qwen35`. The
  public surface is `include/mlx_perf/`; headers in `src/` are internal and
  reach tools and tests only through `mlx_perf_internal` (link it, or a
  support library, never `src/` directly). New sources go into the target
  whose job they do (`CMakeLists.txt`).
- Versions follow `VERSIONING.md`; the MLX releases a version is tested
  against live in `MLX_PERF_TESTED_MLX_VERSIONS`
  (`cmake/mlx_perf_mlx_prefix.cmake`).
- Tests live in `tests/`, one file per production module
  (`<module>_test.cpp`), cases named `should_...`, each module covering
  positive, edge/boundary and negative cases. Run after building:
  `ctest --test-dir build --output-on-failure`. Register a test with
  `mlx_perf_add_test` in `tests/CMakeLists.txt`.
- Model-specific code lives under `src/qwen35/` (config, weights, decode
  step, generation, token text). The generic profiler in `src/` never
  depends on it; `qwen35_profile.cpp` only orchestrates.
- `rust/mlx_perf_rust_example` is the Rust (mlx-bridge) entry point through
  the C API in `include/mlx_perf/mlx_perf_c_api.h`; see its `.cargo/config.toml` for how
  to run it. Build the CMake library first.
- MLX source of truth for API questions: `../mlx-explore/mlx`, checked out at
  0.32.4. The prebuilt prefix may be 0.32.5 (the library builds and tests on
  both), and some backend headers differ between the two (e.g.
  `mlx/backend/metal/device.h`), so confirm a backend detail against the
  prefix's own headers before relying on it.
