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

Runs a two-stream toy model through four passes on freshly built, identical
graphs, prints a job summary, a per-stream table and a per-node Spark-style
latency breakdown, and writes `build/hello_world.html`:

| Pass | What it measures |
|---|---|
| 0. warm-up | discarded — absorbs Metal kernel compilation |
| 1. pipeline | **job wall**: one eval, streams overlap, uninstrumented |
| 2. timeline | per-node issue and host-observed completion with overlap preserved, plus each node's **GPU execution window** read from its Metal command buffer (`GPUStartTime`/`GPUEndTime`) → `wait_on_cpu` / `wait_on_gpu` / `wait_on_dependency` / `scheduler_delay` / `compute` / result latency, and the cross-stream stall (Spark `fetchWaitTime` analog) |
| 3. attributed | **sum-of-stages**: each node synchronized alone |

Every number is printed next to what it is and is not. Per-node `compute` is
GPU-measured at command-buffer granularity (one buffer per node in the
timeline pass); a `~` marks the rare fallback to a host-observed residual
(CPU-stream nodes).

## Reading the report

Open `build/hello_world.html` in any browser (self-contained, works offline,
follows the system light/dark theme). It reads top to bottom like a Spark
StagePage:

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
