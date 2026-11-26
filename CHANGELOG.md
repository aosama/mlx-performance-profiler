# Changelog

Versions follow [`VERSIONING.md`](VERSIONING.md). Each release lists the
MLX releases it was built and tested against.

## Unreleased

### Added
- **Documentation for people and coding agents.** `docs/` holds task
  guides (getting started, profiling a job, comparing two implementations,
  measuring from code, the measurement file, reading the report, building
  and linking, Rust, troubleshooting) and a C API reference, published to
  Context7 through `context7.json` with rules for agents. Every example is
  a file the test suite builds and runs; JSON Schemas for the measurement
  file and the comparison record are checked against real output; and
  `docs_*` tests fail when the docs drift from the code.

## 0.3.0 (2026-11-26)

Tested MLX releases: 0.32.4, 0.32.5 (Metal, macOS).

A minor release under `VERSIONING.md`: the C API and the measurement file
are unchanged, and reports no longer need Graphviz or write an `.svg`.
Consumers pinning `0.2` move to `find_package(mlx_perf 0.3)`.

### Added
- **Interactive execution graph (#30).** The report's DAG is now an
  explorable graph instead of a static Graphviz drawing: nodes coloured by
  where their time went and brighter the more they cost, hover cards with
  the six-slice breakdown, click to trace upstream and downstream, search,
  the hot path (most expensive dependency chain), colour by cost or stream,
  full screen and PNG download. Stage-timed jobs open on their stages and
  drill into each. Built on Cytoscape.js and dagre (MIT), vendored in
  `third_party/` and compiled into the library, so reports stay one offline
  HTML file. The view reads the measurement document (schema version 1).
- **Release pages open with a picture the profiler drew.** The in-repo
  `release-notes` skill captures it from the release's own build
  (`capture_report_visual.py`); the README opens with a gallery of real
  reports.

### Removed
- **Graphviz.** Reports no longer run `dot` or write an `.svg`; the `.dot`
  file is still written. `ReportPaths` loses `svg_path`,
  `graphviz_available` and `plan_svg` (internal C++ only; the C API is
  unchanged).

## 0.2.0 (2026-11-26)

Tested MLX releases: 0.32.4, 0.32.5 (Metal, macOS).

A minor release under `VERSIONING.md`: the C API only gains functions, and
0.1.0 calls keep their signatures and structs. Consumers pinning `0.1`
move to `find_package(mlx_perf 0.2)`.

### Added
- **Per-node measurements as data (#11).** Every profile writes JSON beside
  its report: one record per node in execution order, and a job record with
  every job-wall sample, graph-build time per pass and the command buffers
  observed. Schema version 1, in
  [`docs/measurement-file.md`](docs/measurement-file.md). The new
  `mlx_perf_profile_to_measurements` writes the file alone.
- **Comparing two implementations of one job (#12).**
  `mlx_perf_compare_to_report` (`measure_pair` in C++):
  - checks the two outputs agree;
  - runs 12 uninstrumented rounds, flipping which variant goes first;
  - judges job duration, GPU span and host submission on the per-round
    differences. A result requires the 96% interval for the median
    difference to exclude zero, i.e. at least 10 of 12 rounds, so two
    identical variants are called different under 4% of the time.
  - The comparison page opens with the verdict; `<base>.json` records it.
- **`mlx_perf_set_console_output` (#24)** silences everything the library
  prints.
- Custom Metal kernels are told apart by name (`kernel_name`).

### Changed
- `mlx_perf_profile_to_report` also writes `<base>.json`.
- `hello_world`, `qwen35_profile` and the live server measure eager against
  compiled as an interleaved pair. The live Job tab opens with the verdict
  (#25).
- Serial-pass peak memory is counted above what was already allocated when
  the pass began.
- Consumers building from source see only the C API's headers, and the
  library is optimised even when they set no build type.
- The release workflow refuses a tag that is not on `main`.

### Fixed
- Graphviz runs without a shell, so report and variant names are never
  interpreted as shell syntax.
- NaN or infinite outputs are reported as unmeasurable agreement, not as
  perfect agreement.
- Two variant names differing only in case are refused instead of
  overwriting each other's files.

## 0.1.0 (2026-11-26)

Tested MLX releases: 0.32.4, 0.32.5 (Metal, macOS).

First versioned release of the library:

- C API `mlx_perf_profile_to_report`, `mlx_perf_last_error` and
  `mlx_perf_version`, in `include/mlx_perf/mlx_perf_c_api.h`.
- An installable `libmlx_perf.a` with a `find_package(mlx_perf)` package
  that links MLX from the consumer's own `MLX_PREFIX`.
- The Spark-style HTML report: job summary, per-stream timeline, task
  table and DAG, per node or per job-supplied stage.
