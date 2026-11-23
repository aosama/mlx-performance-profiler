# mlx-visual-plan — Phased Plan: from spike to a Spark-like UX

## 0. Vision

Give an Apple **MLX** computation the **Apache Spark Web UI** experience: capture
the lazy plan, overlay per-node telemetry that separates **computing** from
**waiting** (CPU scheduling, GPU execution, cross-stream dependencies), and
render it the way Spark does — a job summary, a per-stage/task table with a
**stacked latency-breakdown bar**, and a **per-stream Gantt timeline** where
idle gaps are drawn as "stalled".

Big-data-engineer north star: answer *"what did the engine actually do, and what
was it waiting for?"* — the same question Spark's `StagePage` answers for a
shuffle-bound stage.

## 1. Anchoring decisions (recorded now)

- **Language: all C++ in this repo.** Native C++ sidecar linking the prebuilt
  MLX 0.32.4 static library. No rush on the `mlx-c-rust` bridge; an optional
  `Event` binding in `mlx-bridge/src/metal/` is a Phase 6 concern.
- **Keep `graph_model`.** It mirrors MLX's own depth-first traversal with an
  explicit visited set and sibling handling; rewriting it only reintroduces the
  traversal drift this code deliberately guards against.
- **Rewrite `telemetry`.** Keep the serial pass (Spark "sum-of-stages"); add a
  concurrent, `Event`-based pass (Spark "job duration" + stall detection).
- **Rewrite `report_writer`.** Replace the raw DOT/SVG with a Spark-style report
  (summary cards, stacked latency bars, task table, per-stream Gantt).
- **Measurement source of truth.** Host-side concurrent timing + Metal
  `Event` polling for the coarse CPU/GPU/stall story (Phase 1); Metal
  `.gputrace` parsing for kernel-accurate GPU time (Phase 2).
- **Test each module**, per engineering convention (see §7). Keep it lightweight
  through Phase 0–2.

## 2. The Spark decomposition we copy verbatim

From `core/src/main/scala/org/apache/spark/ui/jobs/StagePage.scala`, each task's
wall time is decomposed into seven stacked slices:

```
wall = schedulerDelay + deserialization + shuffleRead(blocked)
     + compute + shuffleWrite + serialization + gettingResult

compute = wall − (all waiting slices)          # the residual
```

Two views are rendered: (a) a **stacked latency bar** per task
(`task-assignment-timeline-duration-bar`), and (b) a **per-task Gantt** on a
wall-clock axis (`vis-timeline`). The canonical "stalled" metric is
`shuffleReadMetrics.fetchWaitTime` — *time blocked waiting for data*. That is
exactly the signal this project is missing, and its MLX analog is a stream
blocked on a cross-stream event (the join).

## 3. Spark slice → MLX measurement

| Spark slice (`StagePage`) | MLX analog | Measurement |
|---|---|---|
| Executor Computing Time (residual) | GPU kernel active time (or CPU-op time) | Metal `Event` / capture timestamps |
| **Shuffle Read Time = *blocked waiting for data*** | A stream **blocked on a cross-stream event** (the join) | GPU-timeline gap: consumer starts after producer's event, with idle time between |
| Shuffle Write Time | Writing intermediates to unified memory | GPU time of the write kernels |
| Task Deserialization | Moving data across streams/threads | host-side copy delta |
| Result Serialization | Copying results back to host | host-side copy delta |
| Scheduler Delay | `eval()` issued → kernel actually enqueued | host-side delta |
| Getting Result Time | Final host fetch | — |

Structural rule copied from Spark: **`compute = total − waiting`**. In MLX:
`GPU_compute = GPU_active − GPU_idle_gaps − cross_stream_waits`.

## 4. Phases

### Phase 0 — Foundations & private control repo
- `git init`, `.gitignore` (`build/`, `*.o`, `.DS_Store`), `README.md`,
  `LICENSE` (propose Apache-2.0 to match Spark + `mlx-bridge`; user's call),
  this `PHASED_PLAN.md`.
- Create the **private** GitHub repo `aosama/mlx-visual-plan` and push.
- **CMake:** make `MLX_PREFIX` a hard requirement with a clear error when unset
  (currently it silently defaults to one cached identity). Document `-DMLX_PREFIX`.
- **Decision recorded here:** keep `graph_model`; rewrite `telemetry` +
  `report_writer`.
- **Acceptance:** a clean clone builds with `-DMLX_PREFIX=…` and
  `./build/hello_world` still emits a report.

### Phase 1 — Non-serializing, GPU-grounded telemetry (reveals stall)
New module `gpu_timeline.{h,cpp}`:
- **Concurrent pass:** run the multi-stream graph **without** per-node full
  `synchronize`; stamp points with `mlx::core::Event` and **poll
  `is_signaled()`** to detect blocking while preserving overlap.
- Per-node host-side duration + per-stream GPU-active windows.
- **Classify each node:** `compute` / `wait_on_gpu` / `wait_on_cpu` /
  `wait_on_dependency`.
- **Cross-stream stall join:** align per-stream windows; a consumer that starts
  after its producer's event, with idle time between, is `wait_on_dependency`.
- Keep the serial pass for the "sum-of-stages" total.
- **Risk (decision needed):** `Event::value()` is a monotonic counter, not wall
  time; converting to ns needs the Metal device clock, which MLX does not
  expose. So Phase 1 reports **host-side CPU/GPU divergence and relative stall**,
  not absolute kernel nanoseconds. Absolute kernel ns arrives in Phase 2 from
  capture. Flag: spike whether a device-clock query is reachable.
- **Acceptance:** the report now shows, per node, a 4-way breakdown and a
  job-level *"sum-of-stages vs job duration"* card.
- **Status (2026-11-23): implemented, console output.** What was learned
  from the MLX 0.32.4 source and is now load-bearing:
  - `array::eval()` **blocks** (`eval_impl(...).wait()`); the non-blocking
    entry point is `async_eval`. Each `async_eval` attaches one Event per
    stream to every array it schedules and signals it when that stream's
    share is done; cross-stream inputs from an earlier call become GPU-side
    event waits. So the timeline pass issues **one `async_eval` per node** in
    topological order (host never blocks, streams overlap) and polls
    `is_available()` for per-node, host-observed completion instants.
  - MLX `detach()`es evaluated arrays (drops inputs + primitive), so the
    dependency snapshot is taken before the first dispatch.
  - Per node, `[issue_started, completed]` is walked causally into
    `wait_on_cpu` (issue) → `wait_on_gpu` (stream busy) →
    `wait_on_dependency` (remote input not ready) → `compute` (residual),
    exactly StagePage's `compute = wall − waits` (`latency_breakdown`).
  - Measured cost of the instrumentation (one command buffer per node) is
    printed as *timeline wall − job wall*; on the toy model ≈ 1–2 ms on a
    ~1–2 ms job, which is why compute here is mostly command-buffer round
    trip and Phase 2 remains necessary for kernel ns.
  - A warm-up pass now precedes the job-wall pass: the first eval in a process
    paid Metal kernel compilation (job wall dropped from ~7 ms to ~1.2 ms).
  - Remaining for this phase's acceptance: surface the breakdown in the HTML
    report (folded into the Phase 3 `report_writer` rewrite).

### Phase 2 — Metal capture ingestion (kernel-accurate GPU time)
- Drive `mlx_metal_start_capture`/`stop_capture` (already wrapped in
  `mlx-bridge`; call from C++ via the C header) around the run with
  `MTL_CAPTURE_ENABLED=1`.
- Parse the `.gputrace`/`sysdetail` for per-kernel GPU start/end/duration.
- Join kernels to graph nodes by op/kernel name.
- Refine the breakdown with true GPU kernel ns; sharpen stall detection to
  kernel granularity (GPU idle gaps = CPU not enqueuing).
- **Risk:** `.gputrace` is a binary Xcode bundle; parsing is non-trivial. Spike
  whether the Instruments `sysdetail` XML or a documented trace schema is
  feasible before committing.
- **Acceptance:** the node table gains a real **GPU time** column; the Gantt GPU
  track shows actual kernels with idle gaps labeled "stalled".
- **Status (2026-11-23): implemented with a different, better-fitting
  mechanism — `.gputrace` parsing was dropped.** Decision and evidence:
  - MLX 0.32.4 exposes `metal::get_command_encoder(stream).get_command_buffer()`
    in its installed headers, and every `async_eval` ends in `gpu::finalize`,
    which commits that buffer and starts a new one. In the timeline pass each
    node is therefore exactly one `MTLCommandBuffer`, whose
    `GPUStartTime`/`GPUEndTime` are GPU-stamped host-timebase seconds.
    `metal_command_buffer_probe` retains the buffer before dispatch and reads
    the window after completion, anchored to the pass clock.
  - Why not `.gputrace`: a binary Xcode bundle with no documented schema,
    needs `MTL_CAPTURE_ENABLED=1` and a capture scope, and capture itself
    perturbs timing. Command-buffer timestamps give per-node GPU time with
    no file format, no environment variable and no MLX change.
  - Why not Metal counter sampling (`MTLCounterSampleBuffer`): per-dispatch
    precision would require inserting sample points inside MLX's own compute
    encoders, i.e. modifying MLX. Deferred; per-node is the granularity the
    report needs.
  - Consequences now in the model: `compute` is GPU-measured (marked `~` when
    it falls back to the host residual, e.g. CPU streams); producer readiness
    uses producers' GPU end; two new slices — `scheduler_delay` (committed →
    GPU start, StagePage's Scheduler Delay, now a fifth class) and
    `completion_latency` (GPU end → host noticed, StagePage's Getting Result
    Time, excluded from classification).
  - Found while building it: a node's Event can be host-visible before Metal
    marks its buffer Completed, so the probe `waitUntilCompleted()`s before
    reading (no accuracy cost — timestamps are GPU-stamped).
  - Finding on the toy model: ~0.35 ms of measured GPU execution inside a
    ~3 ms instrumented pass; nodes are dominated by stream queueing and
    per-buffer submission delay, with real dependency waits at the joins.
  - Limits: a primitive that overflows MLX's per-buffer op budget spans
    several buffers and only the first is seen (underestimate); the
    instrumentation (one buffer per node) adds submission delay a plain eval
    would batch away — the job-wall row keeps that visible.

### Phase 3 — The Spark-style static report (the UX)
Rewrite `report_writer` (keep `dot_writer`/`svg_renderer` as thin layout
helpers):
- **Job summary card:** duration, sum-of-stages vs job duration, peak/active
  memory, stream count.
- **Task table:** op · stream · shape · dtype · bytes · compute · wait_* ·
  memory, sortable (Spark uses DataTables).
- **Stacked latency bar per node:** columns mirror Spark — compute /
  wait_on_cpu / wait_on_gpu / wait_on_dependency / write / copy / serialize.
- **Per-stream Gantt:** GPU track (kernels as bars, gaps = stalled) + CPU track
  (enqueue/host); cross-stream edges as dependency arrows.
- **Color legend** matching Spark's timeline legend (compute green, wait blue,
  stalled red, etc.).
- **Acceptance:** `./build/hello_world` writes an HTML report that reads like a
  Spark stage page for the two-stream toy model.
- **Status (2026-11-23): implemented.** The report is now built from the
  **timeline pass** (measured GPU data), with the serial pass contributing a
  per-node "Serial" column and peak memory; the DAG is labeled with measured
  compute and class. Decisions and changes from the plan:
  - Sections are separate modules (`report_summary_section`,
    `report_gantt_section`, `report_task_table_section`), `report_writer`
    only assembles and saves, and `latency_slices` is the single source of
    slice names, order and colors, so bars, Gantt and legend cannot drift.
  - **No JS library.** Sorting is ~30 lines of inline vanilla JS, and the
    Gantt is server-side SVG generated in C++. A DataTables or vis-timeline
    dependency did not earn its place for a self-contained offline file.
  - The Gantt draws what each stream was *doing* (stalled / scheduling /
    compute), not per-node queued windows: those overlap one another and
    would hide the signal. Queueing stays visible in the bars and the table.
  - "write / copy / serialize" slices from the original list were dropped:
    unified memory has no host↔device copy to measure in this model, so the
    report shows the six slices that are actually measured.
  - Tests: GPU-free component tests render the full document from a
    synthetic fixture (views present, one bar per row, stall arrow, empty
    run with no NaN, hostile labels escaped), plus unit tests for
    `text_format`, `latency_slices` and `report_components`.
  - Finding on the toy model: stream utilization is 2–6%, and about 70% of
    node time is queueing behind per-buffer submission. The GPU waits on
    submission order, not on math.

### Phase 4 — Compiled vs uncompiled view
- `compile()` the same closure; render the **fused tape** next to the eager plan
  (Spark "planned vs executed"). Show which primitives MLX fused and how that
  changes per-node timing.
- **Acceptance:** side-by-side eager vs compiled graphs with a timing-delta
  table.
- **Status (2026-11-23): implemented.** `hello_world` measures the same job
  twice (eager, and through `mlx::core::compile`) and writes
  `hello_world_eager.html`, `hello_world_compiled.html`, and the comparison
  page `hello_world.html`: metric deltas, both DAGs side by side, both
  timelines on one shared time scale, and a primitive mix that names what
  each `Compiled…` kernel fused (MLX names it `Compiled` + the fused ops).
  Decisions and changes:
  - **The demo job changed.** Inputs are materialized up front, and the
    measured job is a pure three-stream function (`demo_model`). Before, about
    70 of 81 nodes were RNG plumbing, which drowned the signal, and `compile`
    needs a pure function. The job is sized for real GPU work (a
    256×1024×1024 addmm and an elementwise chain over 2M elements).
  - `measurement_session` runs the four passes per variant behind a
    `GraphFactory` interface, so main only orchestrates. Its result is
    move-only because timelines point into the captured graphs.
  - **The job wall is now the median of 7 runs, observed by polling.** A
    single blocking `eval()` swung ~1 ms on a ~1 ms job (OS wake-up
    latency), sometimes making the instrumented pass look faster than the
    uninstrumented one. The job pass now uses one `async_eval` and spins on
    `is_available()`, the same observer as the timeline pass, and reports
    the min–max range. When the instrumented wall falls inside that range,
    the report says "no measurable instrumentation cost" instead of
    printing noise as a number.
  - Compile never crosses streams (`compile.cpp` stops fusion at a stream
    boundary), so both variants keep the same stream structure and compare
    fairly. A test asserts eager and compiled outputs are `allclose`.
  - Finding: compile halves executed nodes (18 → 9) and roughly halves job
    duration; the 5-kernel elementwise chain becomes one ~270 µs kernel, and
    the cross-stream stall at the join shrinks with it.

### Phase 5 — Live, interactive server
- Replace the static HTML with a small local server (Spark-UI-like tabs: Job /
  Stages / Tasks / Memory), streaming results.
- **Acceptance:** `mlx_plan_server` serves the report at `localhost:PORT` with
  the same views as Phase 3, live.
- **Status (2026-11-23): implemented.** `mlx_plan_server [--port N]`
  serves Job / Stages / Tasks / Memory tabs, a Re-run button and a live
  status line. Decisions:
  - **Threading.** All MLX work (measuring, rendering, memory sampling)
    stays on the main thread, because MLX's default streams and Metal
    encoders are per-thread. HTTP threads only read `LiveViewStore`, which
    holds plain data: rendered HTML strings, memory samples, a status line
    and run requests.
  - **No web framework.** A ~200-line POSIX-socket HTTP/1.1 server
    (`http_server`), one thread per connection, **bound to 127.0.0.1 only**,
    with an 8 KiB request-head limit (431 when exceeded). Live updates use
    Server-Sent Events; tabs reload only when the rendered views change, so
    a reader's sort order survives status ticks.
  - **Views reuse the static report sections** (`live_view_renderer`).
    Memory is the one new view: per-node memory added (serial pass) and a
    1 Hz chart of active, peak and cache memory.
  - Configuration is a `--port` flag, never an environment variable.
    SIGINT/SIGTERM close event streams and stop cleanly.
  - Tests (no GPU): `http_server` over real sockets (malformed → 400,
    oversized → 431, stop is idempotent), `live_view_store` (versions,
    sample cap, run hand-off between threads), `live_view_routes` (every
    route, JSON escaping, SSE initial update, 404/405), `server_options`,
    and `report_memory_section`.
  - Verified by hand: curl over every route, SSE updates during a
    triggered re-run, a browser render, and a clean SIGINT stop with an
    open stream.

### Phase 6 — Rust bridge integration (optional `Event` binding)
- Re-express capture via `mlx-c-rust` where the C API allows; optionally add an
  `Event` binding in `mlx-bridge/src/metal/` (alongside `capture.rs`) so the view
  is reachable from the bridge ecosystem.
- **Acceptance:** an `mlx-c-rust` example emits the same telemetry.
- **Status (2026-11-23): implemented without modifying `mlx-bridge`.**
  Evidence and decisions:
  - MLX-C exposes `mlx_async_eval`, `_mlx_array_is_available`, memory
    stats and Metal capture, but **not** graph traversal (inputs, primitive
    name, stream) or Metal command buffers, which the telemetry depends on.
    Re-implementing the telemetry in Rust would mean a large new C shim in
    the bridge plus duplicated logic that drifts. Instead, this repo exposes
    its existing C++ measurement through a small **C ABI**
    (`mlx_plan_c_api.h`): the caller passes a callback that appends a fresh
    graph's outputs to an `mlx_vector_array`, and the library runs the same
    four passes and writes the same report.
  - Ownership never crosses the boundary (the library creates and frees the
    vector; MLX-C copies handles on append), and exceptions never cross it
    either (status code + `mlx_plan_last_error()`). The one private-layout
    assumption (an MLX-C `mlx_array`'s `ctx` is an `mlx::core::array*`)
    lives in `mlx_c_interop` and is pinned by a test.
  - **One MLX in the binary.** The bridge links the same shared prebuilt
    prefix, and `libmlx_plan.a` holds only this repo's objects, so every MLX
    symbol resolves to the bridge's single `libmlx.a`.
  - `rust/mlx_plan_rust_example` (pinned to the bridge's nightly) builds a
    three-stream job with the safe API (`MlxBindingsContext::on_stream`)
    and profiles it: 19/19 nodes GPU-measured, three streams, a measured
    cross-stream stall, and `build/rust_example.html`. Verified against the
    local bridge checkout at `d54c0d9`.
  - The optional `Event` binding in `mlx-bridge` is **not needed** and was
    not added; no bridge PR was required. Upstreaming graph traversal to
    MLX-C would let a pure-Rust implementation exist later; it is not a
    prerequisite for using this tool from Rust.
  - Found while building it: C++ progress lines were fully buffered when
    stdout is a pipe (Rust parent, `tee`, CI), so "live" progress arrived
    at exit. Progress and pass announcements now flush. Also,
    `write_report` silently ignored write failures; `ReportPaths` now has
    `is_written`.

## 5. Module map at the end state

| Module | Job | When |
|---|---|---|
| `graph_model` | Walk the graph into nodes/edges/topo order (the captured plan) | keep |
| `telemetry` | Serial pass — sum-of-stages total | keep + extend |
| `gpu_timeline` | Overlap-preserving per-node pass (one `async_eval` per node) | P1 ✓ |
| `latency_breakdown` / `latency_slices` | StagePage decomposition; slice vocabulary | P1–P3 ✓ |
| `metal_command_buffer_probe` | Per-node GPU window from command-buffer timestamps (replaced `capture_ingestion`) | P2 ✓ |
| `report_*` sections + `report_writer` | Spark-style HTML (summary, Gantt, task table, DAG) | P3 ✓ |
| `text_format` | Shared units and escaping | P3 ✓ |
| `dot_writer` / `svg_renderer` | Graphviz layout helpers | keep, thin |
| `demo_model` / `graph_factory` / `measurement_session` | The job, and the four passes per variant | P4 ✓ |
| `report_comparison_page` | Eager vs compiled side by side | P4 ✓ |
| `http_server` / `live_view_*` / `server_options` / `report_memory_section` | Live server | P5 ✓ |
| `hello_world` / `mlx_plan_server` | Orchestration only | ✓ |

## 6. Testing strategy (per engineering convention)

- **Unit:** `graph_model` traversal correctness; `gpu_timeline` node
  classification; `capture_ingestion` kernel→node join.
- **Integration:** end-to-end `hello_world` produces a valid report with the
  expected totals.
- **Golden:** DOT/HTML golden files for the toy model (regression guard).
- **Perf / sanity:** assert `sum_of_stages ≥ job_duration` (decomposition
  invariant) and no regression in run time.
- **Acceptance:** a big-data engineer can answer *"what was stalled"* from the
  report alone.
- Keep it lightweight: an assert-based runner or GoogleTest via FetchContent;
  no heavy deps in Phase 0–2.

## 7. Open decisions / risks

- **`Event` value → time conversion** (Phase 1 risk): monotonic counter, not wall
  time; MLX does not expose the Metal device clock.
- **`.gputrace` parsing feasibility** (Phase 2 spike): binary Xcode bundle.
- **License:** propose Apache-2.0; user's call.
- **`MLX_PREFIX` default vs hard error** (Phase 0).
- **Always show both totals honestly** (sum-of-stages vs job duration) — the
  serial measurement changes semantics, exactly as Spark separates stage
  durations from job duration. Never present one as the whole truth.
