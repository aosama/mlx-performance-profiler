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

### Phase 4 — Compiled vs uncompiled view
- `compile()` the same closure; render the **fused tape** next to the eager plan
  (Spark "planned vs executed"). Show which primitives MLX fused and how that
  changes per-node timing.
- **Acceptance:** side-by-side eager vs compiled graphs with a timing-delta
  table.

### Phase 5 — Live, interactive server
- Replace the static HTML with a small local server (Spark-UI-like tabs: Job /
  Stages / Tasks / Memory), streaming results.
- **Acceptance:** `mlx_plan_server` serves the report at `localhost:PORT` with
  the same views as Phase 3, live.

### Phase 6 — Rust bridge integration (optional `Event` binding)
- Re-express capture via `mlx-c-rust` where the C API allows; optionally add an
  `Event` binding in `mlx-bridge/src/metal/` (alongside `capture.rs`) so the view
  is reachable from the bridge ecosystem.
- **Acceptance:** an `mlx-c-rust` example emits the same telemetry.

## 5. Module map at the end state

| Module | Job | When |
|---|---|---|
| `graph_model` | Walk the graph into nodes/edges/topo order (the captured plan) | keep |
| `telemetry` | Serial pass — sum-of-stages total | keep + extend |
| `gpu_timeline` | Concurrent `Event` pass + stall classification | new (P1) |
| `capture_ingestion` | Metal trace parsing → kernel-accurate GPU time | new (P2) |
| `report_writer` | Spark-style HTML (summary, bars, table, Gantt) | rewrite (P3) |
| `dot_writer` / `svg_renderer` | Graphviz layout helpers | keep, thin |
| `hello_world` / `server` | Orchestration only | keep / extend |

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
