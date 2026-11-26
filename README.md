# mlx-performance-profiler

A feasibility experiment: give Apple [MLX](https://github.com/ml-explore/mlx)
computations the Apache Spark Web UI experience. It captures MLX's lazy
computational graph, measures what each node spent **computing** versus
**waiting** (on the host, on its own GPU stream, or on another stream's data),
and reports it the way Spark's StagePage does.

It is an observer: it never changes MLX semantics.

Formerly named `mlx-visual-plan`; the project is a profiler, and the name now
says so. In code it uses the short prefix `mlx_perf`.

See [`PHASED_PLAN.md`](PHASED_PLAN.md) for the roadmap and
[`AGENTS.md`](AGENTS.md) for contributor conventions.

## Build

Requires CMake ≥ 3.24, a C++20 compiler, and a prebuilt MLX 0.32.4
installation prefix (containing `include/` and `lib/`). `MLX_PREFIX` is a
required CMake cache variable; it is never read from the environment.

```sh
cmake -S . -B build -DMLX_PREFIX=/path/to/mlx/prefix
cmake --build build
```

Graphviz (`dot`) is optional: with it the HTML report embeds an SVG of the
plan, without it the report shows the raw DOT source.

## Run

```sh
./build/hello_world
```

Measures one small three-stream job twice, **eager** and through
`mlx::core::compile`. Each variant goes through four passes on freshly built,
identical graphs. It prints a job summary, a per-stream table and a per-node
latency breakdown per variant, and writes three reports:

- `build/hello_world.html`: **eager vs compiled** side by side (metric
  deltas, both DAGs, both timelines on one time scale, and which primitives
  compile fused).
- `build/hello_world_eager.html` and `build/hello_world_compiled.html`: the
  full Spark-style report for each variant.

The four passes per variant:

| Pass | What it measures |
|---|---|
| 0. warm-up | discarded — absorbs Metal kernel compilation |
| 1. pipeline | **job wall**: median of 7 uninstrumented runs (one `async_eval`, streams overlap), with the min–max range |
| 2. timeline | per-node issue and host-observed completion with overlap preserved, plus each node's **GPU execution window** read from its Metal command buffer (`GPUStartTime`/`GPUEndTime`) → `wait_on_cpu` / `wait_on_gpu` / `wait_on_dependency` / `scheduler_delay` / `compute` / result latency, and the cross-stream stall (Spark `fetchWaitTime` analog) |
| 3. attributed | **sum-of-stages**: each node synchronized alone |

Every number is printed next to what it is and is not. Per-node `compute` is
GPU-measured at command-buffer granularity (one buffer per node in the
timeline pass); a `~` marks the rare fallback to a host-observed residual
(CPU-stream nodes).

## Live view

```sh
./build/mlx_perf_server            # or: --port 9000
```

Open <http://127.0.0.1:8765/>. The server measures the job (eager and
compiled) and serves Spark-UI-like tabs:

- **Job:** the comparison table plus each variant's summary cards.
- **Stages:** per-stream Gantt and stream table.
- **Tasks:** the sortable per-node table.
- **Memory:** a live process-memory chart sampled at 1 Hz, plus the nodes
  that added the most memory.

**Re-run** measures again. The status line and tabs update over
Server-Sent Events, so there's no page refresh. The server listens on
127.0.0.1 only; Ctrl-C stops it.

### Previewing inside the Claude desktop app

[`.claude/launch.json`](.claude/launch.json) tells the Claude desktop app
(Code tab) how to start two local servers in its built-in browser pane, so
Claude can open and click through the views with you:

| Name | Port | What it serves |
|---|---|---|
| `mlx-perf-live` | 8765 | the live view above (`./build/mlx_perf_server`) |
| `mlx-perf-reports` | 8766 | the static HTML reports in `build/` (e.g. `/hello_world.html`, `/qwen35_decode.html`), via `python3 -m http.server` |

The reports server exists because the browser pane does not open local
`file://` pages. Both bind to 127.0.0.1 only. Build first (see **Build**):
the live server needs `build/mlx_perf_server`, and the reports server needs
reports already written to `build/`. Outside the app, run the same commands
in a terminal; the file is ignored by everything else.

## Profiling a real model (Qwen3.5)

```sh
./build/qwen35_profile /path/to/Qwen3.5-0.8B-MLX-4bit            # or: --context 4096
```

This loads a dense Qwen3.5 MLX checkpoint (4- or 8-bit; the bit width is
derived per layer). It rebuilds the whole model in C++ from MLX's production
kernels: quantized matmul, RMSNorm, RoPE, fused attention, and the fused
Gated DeltaNet update.

It **first proves the math**: greedy decoding of "The capital of France is"
must continue with " Paris", or the tool refuses to profile. Then it measures
one decode step at the given context length, eager vs compiled, and writes
`build/qwen35_decode.html` plus a report per variant.

### Stage granularity

The timeline pass issues one `async_eval` per timed unit, and each one
commits at least one Metal command buffer. At about 1,100 nodes per step,
one buffer per node costs several times the real step (eager: 24.2 ms
instrumented for a 3.1 ms step), so nearly every node reads as queued behind
the instrumentation. The profiler therefore times **stages** when the job names
them and the graph is large (≥ 256 primitives). Here a stage is one decoder
layer (Spark's stage), and it costs one `async_eval`:

| Qwen3.5-0.8B-4bit, context 1024 | job wall | per-node timeline | per-stage timeline |
|---|---|---|---|
| eager (1,144 nodes, 25 stages) | 3.15 ms | 24.19 ms (7.7×) | 3.61 ms (1.14×) |
| compiled (766 nodes, 25 stages) | 2.92 ms | 11.29 ms (3.8×) | 3.06 ms (1.05×) |

With the instrumentation out of the way, the waiting breakdown means
something again. The host takes about 80–210 µs to encode each eager layer,
and the GPU takes about 75–145 µs to run it. So the second half of the eager
step is classed `wait_on_cpu` (host-bound encoding), while the first half
and most compiled stages are `queued` (the host ran ahead of the GPU). The
LM head (a 248k-row quantized matmul) is the largest single stage, at
~0.55 ms.

```sh
./build/qwen35_profile /path/to/model --granularity node     # force per-node (or: stage, auto)
```

What a stage row measures: its GPU window runs from the GPU start of the
stage's first command buffer to the end of its last. MLX commits mid-stage
when a buffer exceeds its op budget, and the probe cannot see those middle
buffers. The end is therefore taken from an empty marker buffer queued
behind the stage on the same Metal queue, tightened by host-observed
completion. Compute includes any GPU gap while the host was still encoding
the stage. Members of a stage finish together, so per-node waiting is not
observable in this mode; per-node detail stays the default for small graphs.
A job supplies stages by overriding `GraphFactory::build_fresh_staged_graph`.
The profiler itself knows nothing about Qwen.

## From Rust (mlx-c-rust)

The same measurement and report are available through a C API
(`src/mlx_perf_c_api.h`). [`rust/mlx_perf_rust_example`](rust/mlx_perf_rust_example)
builds a three-stream job with `mlx-c-rust` and profiles it:

```sh
cmake --build build                      # builds libmlx_perf.a
cd rust/mlx_perf_rust_example
cargo profile-job                        # writes build/rust_example.html
```

It expects the `mlx-bridge` checkout as a sibling directory and uses that
crate's pinned nightly toolchain. The final binary links exactly one MLX: the
static `libmlx_perf.a` contains only this repository's code, and it resolves
MLX from the same prebuilt prefix the bridge links.

## Reading the report

Open `build/hello_world.html` for the comparison, or a variant report, in any
browser. Each is self-contained, works offline and follows the system
light/dark theme. A variant report reads top to bottom like a Spark StagePage:

1. **Job summary.** Cards for job duration (the honest end-to-end number),
   measured GPU execution, cross-stream stall, instrumented duration, sum of
   stages and peak memory. Under them, **where node time went** is one stacked
   bar over all nodes.
2. **Timeline (per stream).** A Gantt with one lane for the host thread that
   issues nodes and one lane per MLX stream. On a stream lane, **red = stalled**
   on another stream's data, **amber = GPU scheduling**, **green = measured GPU
   execution**, and blank means nothing was runnable. Arrows run from a
   producer's GPU end to where its cross-stream consumer stopped waiting.
   Below it is a per-stream table with busy, stalled and utilization.
3. **Tasks.** One sortable row per executed node: each slice, wall, the same
   node timed alone ("Serial"), its class, and a stacked latency bar. In a
   stage-timed report, one row per stage and stream instead, with the ops
   inside it and the members' serial times summed.
4. **DAG.** The plan graph: fill = stream, red outline = node stalled on
   another stream, edges labeled with bytes. Stage-timed reports box each
   stage as a dashed cluster, as Spark's DAG view does.

The six slices, in causal order, always sum to a node's wall:

| Slice | Meaning | Spark analog |
|---|---|---|
| host issue | host time inside `async_eval` | Scheduler Delay (host) |
| queued | earlier work on the same stream still running | — |
| stalled | input from another stream not ready yet | Shuffle Read `fetchWaitTime` |
| GPU scheduling | committed, GPU not started yet | Scheduler Delay |
| compute | GPU execution of the node's command buffer | Executor Computing Time |
| result | GPU done → host noticed (never decides the class) | Getting Result Time |

## Test

```sh
ctest --test-dir build --output-on-failure
```

`latency_breakdown_test` and `stage_plan_test` are GPU-free unit tests of
the decomposition and the stage grouping; `gpu_timeline_test` and
`stage_timeline_test` are integration tests that run on the Metal GPU.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
