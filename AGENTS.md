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
- C++20, one job per file: graph capture, telemetry, DOT writing, the graph
  view section, and HTML report writing are separate modules with separate
  headers. `main`-level files only orchestrate.
- Kernel launches are observed by `metal_dispatch_recorder`: Objective-C
  runtime hooks on the concrete classes of MLX's Metal compute encoder and
  device, which call the original first and record only inside a
  `DispatchRecordingScope` on the calling thread. Keep them observation-only;
  `dispatch_occupancy` holds the pure arithmetic, `gpu_device_facts` the
  core count.
- The interactive graph view is JavaScript in `src/graph_view/` (model,
  style, panels, mounting: one job per file), embedded into the library at
  build time with `third_party/`'s vendored libraries
  (`cmake/mlx_perf_embed_text_assets.cmake`). It reads the measurement
  document (schema version 1), never a format of its own. Its model, style and
  panels are unit-tested under Node: `node --test tests/graph_view/`. To update a
  vendored library, follow `third_party/README.md`.
- A single function stays around 40 lines; split at natural seams when longer.
- Comments carry rationale (why this design, what tradeoff), not narration.
- Variable names describe domain meaning: `output_node_ids`,
  `stream_index_to_stream`, `node_id_to_telemetry`. No `tmp`/`data`/`info`.
- Any run longer than a few seconds prints live progress with an ETA.
- Library code prints only through `console_output` (`print_progress_line`,
  `print_error_line`), never `printf`/`std::cout` directly: callers switch
  all of it off with `mlx_perf_set_console_output(0)`. Tools' own `main`
  files may print directly.
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
- Documentation lives in `docs/` (indexed by Context7 through
  `context7.json`): one question per page, with complete examples. Every C,
  CMake and Python example is a file in `docs/examples/`, copied into the
  pages verbatim between `<!-- example: <path> -->` markers; edit the file,
  then run `python3 tests/docs/docs_check.py sync .`. CTest's `docs_*` tests
  fail when a copy drifts, a link or anchor breaks, the C API reference
  misses a header declaration, troubleshooting quotes a message the code
  cannot produce, a release pin names another version, `context7.json`
  breaks Context7's limits, or a real measurement file disagrees with
  `docs/schema/`. A new JSON field needs its schema entry; a new error
  message needs its troubleshooting entry.
- Releases: write `release-notes/vX.Y.Z.md` with the `release-notes` skill
  (the release skill, `.claude/skills/release-notes/`) in the release pull
  request; the release workflow publishes it and refuses a tag without it.
  The same skill's step 6 makes sure `docs/` and `context7.json` cover every
  new capability and changed behaviour, and that Context7 serves them after
  the merge.
- Versions follow `VERSIONING.md`; the MLX releases a version is tested
  against live in `MLX_PERF_TESTED_MLX_VERSIONS`
  (`cmake/mlx_perf_mlx_prefix.cmake`).
- Tests live in `tests/`, one file per production module
  (`<module>_test.cpp`), cases named `should_...`, each module covering
  positive, edge/boundary and negative cases. Run after building:
  `ctest --test-dir build --output-on-failure`. Register a test with
  `mlx_perf_add_test` in `tests/CMakeLists.txt`.
- Procedure comparisons (#38) measure code that evaluates many graphs:
  `metal_submission_recorder` hooks MLX's command buffer commits and
  blocking waits (observation-only, like `metal_dispatch_recorder`);
  `procedure_measurement` runs the rounds; the page, the record and the
  files are separate modules (`report_procedure_page`,
  `procedure_comparison_record`, `procedure_report_files`). The C entry
  point is `mlx_perf_c_api_procedures.cpp`; plumbing every entry point
  shares is in `c_api_support`.
- Wired memory (#45): `system_wired_memory` reads the kernel (sysctl, not
  the rate-limited `host_statistics64`); `wired_memory_sample` and
  `wired_memory_analysis` are pure; `wired_memory_sampler` takes samples
  (an `EvaluationBoundaryListener` the submission recorder calls after each
  blocking wait); `wired_memory_record` and `report_wired_memory_section`
  present them; `wire_release_probe` is the probe. Sample only runs whose
  timing is not judged.
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
