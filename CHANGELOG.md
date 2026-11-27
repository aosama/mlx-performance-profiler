# Changelog

Versions follow [`VERSIONING.md`](VERSIONING.md). Each release lists the
MLX releases it was built and tested against.

## Unreleased

### Added
- **Linux with MLX's CUDA backend (#15).** The profiler builds against an
  MLX built with `-DMLX_BUILD_CUDA=ON` and measures NVIDIA GPUs through
  CUPTI, NVIDIA's observation interface: one CUPTI session
  (`cupti_session`) feeds a CUDA execution probe (each timed unit's GPU
  window from the start of its first kernel to the end of its last, over
  every CUDA graph launch it made), a CUDA dispatch recorder (each kernel's
  name, grid, block and shared memory, read as MLX adds it to its CUDA
  graph) and a CUDA submission recorder (graph launches and the host's
  blocking waits). The GPU backend is detected from the MLX prefix at
  configure time; the Metal implementation is unchanged behind the same
  interfaces. Reports and console output name what each backend measures
  (CUDA kernels, thread blocks, streaming multiprocessors, CUPTI
  timestamps). Wired memory and the macOS memory pressure counters have no
  Linux counterpart and read as unavailable.

### Changed
- **`schema_version` 2.** `kernel_dispatches[].metal_function_name` and the
  procedure record's `kernel_launches[].metal_function_name` are renamed
  `kernel_function_name`, since on CUDA they hold a CUDA kernel's name; the
  top-level `backend` is `"metal"` or `"cuda"`. Readers of version 1 files
  need the rename only.
- **Backend-neutral internal headers.** `metal_dispatch_recorder.h`,
  `metal_submission_recorder.h` and `metal_host_clock.h` become
  `dispatch_recorder.h`, `submission_recorder.h` and `gpu_host_clock.h`
  (`current_gpu_host_seconds`), and the timeline passes use
  `GpuExecutionProbe`; each has one implementation per backend.

### Verified
- **Every tested MLX release, on the current code.** The release skill
  gains a step that runs the full suite on each release in
  `MLX_PERF_TESTED_MLX_VERSIONS` before a release, with the skill's
  `find_mlx_prefix.py` checking each prefix's MLX version. Run on the code
  after 0.6.0: all 90 tests pass on MLX 0.32.4 and on MLX 0.32.5. Until now
  0.32.4 had last been run on the 0.3.0 code.

## 0.6.0 (2026-11-26)

Tested MLX releases: 0.32.4, 0.32.5 (Metal, macOS).

A minor release under `VERSIONING.md`: the C API gains a struct and
`mlx_perf_procedure_comparison_summary` gains fields (it grows from 496 to
704 bytes, so code that allocates it must be rebuilt); the JSON files gain
fields under the same `schema_version` 1. Consumers pinning `0.5` move to
`find_package(mlx_perf 0.6)`.

### Added
- **Memory judged like time (#46).** Procedure and graph comparisons judge
  three memory figures on the same paired rounds as their times, with the
  same interval and verdict ("less" or "more"): each run's peak MLX working
  set (MLX's peak active memory less its active memory as the run began),
  what it left behind (MLX active + cache, end less start), and the
  system's wired total at its end. They are read just before and just
  after each run, never during it, so the timing pays nothing for them.
  A note under the timing verdict counts the rounds in which either side
  crossed the wire limit (the kernel refused wire requests over it, or the
  wired total ended over it) or saw the system compress memory, since such
  rounds are slow for a reason outside the code. Every round's figures are
  in the JSON (`rounds[].memory`; the comparison record's `memory`), from
  live sysctl counters (`vm.add_wire_count_over_global_limit`,
  `vm.compressor_input_bytes`, `vm.swapusage`).
- **C API:** `mlx_perf_memory_comparison`, and
  `mlx_perf_procedure_comparison_summary` gains `peak_mlx_active_added`,
  `mlx_held_change`, `system_wired_at_end` and four round counts. The struct
  grows from 496 to 704 bytes: rebuild code that allocates it.
  `mlx_perf_comparison_summary` (graph comparisons) is unchanged, so its
  Rust mirror stays valid; read those figures from the JSON.

## 0.5.0 (2026-11-26)

Tested MLX releases: 0.32.4, 0.32.5 (Metal, macOS).

A minor release under `VERSIONING.md`: the C API gains functions and
structs without changing existing ones, and the measurement file and
procedure comparison record gain fields under the same `schema_version` 1.
Consumers pinning `0.4` move to `find_package(mlx_perf 0.5)`.

### Added
- **Wired memory beside MLX's own (#45).** On Apple silicon the GPU's memory
  is wired and macOS keeps wired memory under `vm.global_user_wire_limit`.
  Reports gain a **Wired memory** lane sampled after every node of the
  serial pass; procedure comparisons gain one per procedure, sampled over
  its observed run (not the judged rounds) after every blocking wait on any
  thread, every millisecond, and after the run until macOS un-wired what it
  released. Each shows the wired total, MLX's active memory and cache, the
  part wired outside MLX, the limit and the room under it, every MLX release
  with how long macOS took to un-wire it, and the samples whose "outside
  MLX" that delay inflated. The JSON carries the same as `job.wired_memory`
  and `baseline` / `candidate.wired_memory` (fields added; schema version 1).
  New C API: `mlx_perf_read_wired_memory` and
  `mlx_perf_measure_wire_release_lag`, with `mlx_perf_wired_memory` and
  `mlx_perf_wire_release_lag`. Measured on one Mac: released memory stays
  wired about 0.2 s; memory MLX still holds is un-wired after about 1.2 s
  without GPU use. The wired total is read through `sysctl
  vm.page_wired_count`, because macOS rate limits `host_statistics64`.
- **Byte sizes of 1 GiB and over read in GiB** on every page (for example
  "37.92 GiB" instead of "38830.6 MiB").

### Fixed
- **Procedures that evaluate on another thread (#44).** A procedure whose
  `run` hands its MLX work to a worker thread (an engine with an owner
  thread for MLX) was reported with no command buffers, no GPU busy time
  and the whole run as GPU idle. Every command buffer, blocking wait and
  kernel launch anywhere in the process while `run` runs is now counted as
  the procedure's: procedures run one at a time, so it is theirs. Graph
  jobs' serial pass still records only the evaluating thread.

## 0.4.0 (2026-11-26)

Tested MLX releases: 0.32.4, 0.32.5 (Metal, macOS).

A minor release under `VERSIONING.md`: the C API gains functions and
structs without changing existing ones, and the measurement file gains
fields under the same `schema_version` 1. Consumers pinning `0.3` move to
`find_package(mlx_perf 0.4)`.

### Verified
- **MLX 0.32.4.** Until now every test run used MLX 0.32.5, though 0.32.4 was
  declared. The 0.3.0 library code now passes the full suite (57 tests,
  including the Metal GPU, packaging and documentation-example tests) on
  MLX 0.32.4, and a Qwen3.5 decode step profiles the same 1,144 nodes in 25
  stages there.

### Added
- **Kernel launch shapes and GPU coverage (#35).** Every kernel launch MLX
  encodes is recorded in the serial pass: the Metal function, the grid and
  threadgroup sizes, the threadgroups formed, the pipeline's limits and its
  threadgroup memory, and the share of GPU cores it can give work to (from
  the GPU's core count). A launch with fewer threadgroups than cores is
  flagged. It appears in the measurement file (`nodes[].kernel_dispatches`,
  `job.gpu`; additive, schema version 1), the task table, the graph view
  (a warning line, a GPU coverage colour mode, hover details) and a new
  "Kernel launches side by side" table on the comparison page. The hooks
  observe MLX's Metal encoder without changing it, and record only while
  the serial pass evaluates a node.
- **Procedure comparisons (#38).** `mlx_perf_compare_procedures_to_report`
  compares two procedures, code that evaluates any number of MLX graphs (a
  token loop, a sampler, a pipeline), each measured whole: a `run` callback
  timed until every command buffer it committed has completed, and an
  optional untimed `prepare` before every run. The same 12 interleaved
  rounds and verdict judge duration, GPU busy time (the union of its
  command buffers' GPU windows) and GPU idle time. Observation-only hooks on
  MLX's Metal command buffers and shared events count each run's command
  buffers and blocking waits. It writes a page (verdict, hand-offs per run,
  kernel launches by Metal function) and a record (`"kind":
  "procedure_comparison"`, schema version 1, with its own JSON Schema).
  New C API: `mlx_perf_procedure`, `mlx_perf_procedure_step_fn`,
  `mlx_perf_procedure_summary`, `mlx_perf_procedure_comparison_summary`.
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
