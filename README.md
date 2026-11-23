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

## Test

```sh
ctest --test-dir build --output-on-failure
```

`latency_breakdown_test` is a GPU-free unit test of the decomposition;
`gpu_timeline_test` is an integration test that runs on the Metal GPU.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
