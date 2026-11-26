# Instructions for maintaining mlx-performance-profiler

## What this repo is

An experiment: reproduce the useful parts of Apache Spark's Web UI
("what did the engine actually do") for Apple MLX computations, by observing
MLX's lazy computational graph and adding the telemetry Spark shows per stage.

The engine API surface lives in the sibling project `../mlx-bridge`
(the `mlx-c-rust` Rust bridge over MLX-C). This repo is the profiling
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
- Tests live in `tests/`, one file per production module
  (`<module>_test.cpp`), cases named `should_...`, each module covering
  positive, edge/boundary and negative cases. Run after building:
  `ctest --test-dir build --output-on-failure`.
- Model-specific code lives under `src/qwen35/` (config, weights, decode
  step, generation, token text). The generic profiler in `src/` never
  depends on it; `qwen35_profile.cpp` only orchestrates.
- `rust/mlx_perf_rust_example` is the Rust (mlx-c-rust) entry point through
  the C API in `src/mlx_perf_c_api.h`; see its `.cargo/config.toml` for how
  to run it. Build the CMake library first.
- MLX source of truth for API questions: `../mlx-explore/mlx` (same version
  0.32.4 as the prebuilt library; headers in the prefix and the source tree
  are in sync).
