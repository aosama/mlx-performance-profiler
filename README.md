# mlx-visual-plan

A feasibility experiment: give Apple [MLX](https://github.com/ml-explore/mlx)
computations the Apache Spark Web UI experience. It captures MLX's lazy
computational graph, measures what each node spent **computing** versus
**waiting** (on the host, on its own GPU stream, or on another stream's data),
and reports it the way Spark's StagePage does.

It is an observer: it never changes MLX semantics.

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
./build/mlx_plan_server            # or: --port 9000
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

## From Rust (mlx-c-rust)

The same measurement and report are available through a C API
(`src/mlx_plan_c_api.h`). [`rust/mlx_plan_rust_example`](rust/mlx_plan_rust_example)
builds a three-stream job with `mlx-c-rust` and profiles it:

```sh
cmake --build build                      # builds libmlx_plan.a
cd rust/mlx_plan_rust_example
cargo profile-job                        # writes build/rust_example.html
```

It expects the `mlx-bridge` checkout as a sibling directory and uses that
crate's pinned nightly toolchain. The final binary links exactly one MLX: the
static `libmlx_plan.a` contains only this repository's code, and it resolves
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
   node timed alone ("Serial"), its class, and a stacked latency bar.
4. **DAG.** The plan graph: fill = stream, red outline = node stalled on
   another stream, edges labeled with bytes.

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

`latency_breakdown_test` is a GPU-free unit test of the decomposition;
`gpu_timeline_test` is an integration test that runs on the Metal GPU.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
