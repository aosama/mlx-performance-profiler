# mlx-visual-plan

A feasibility experiment: can an Apple **MLX** computation be given the kind of
**Apache Spark Web UI** experience — "here is the plan you asked for, and here
is what the engine actually did, with durations and bytes overlaid"?

Answer so far: **yes**, with one caveat. MLX's C++ API already provides the
plan capture (a lazy computational graph traversable from output arrays, plus a
built-in DOT exporter). What it does not provide is per-node execution timing,
so this repo manufactures it by evaluating the graph **node by node in
topological order** with a stream synchronize around each node — the Spark
"stage duration" analog.

## The Spark → MLX mapping this repo implements

| Spark Web UI concept                | MLX equivalent used here                                      |
| ----------------------------------- | -------------------------------------------------------------- |
| Physical plan tree (`EXPLAIN`)      | DFS from output arrays through `array::inputs()`/`primitive()` |
| Task / stage node                   | One MLX primitive application (one `array` with a primitive)   |
| Stage parallelism                   | `Stream` — every `Primitive` carries the stream it runs on     |
| Shuffle bytes                       | `input.nbytes()` on every graph edge                           |
| Per-stage duration bar              | Wall-clock around per-node `eval()` + `synchronize(stream)`    |
| Peak / active memory counters       | `get_peak_memory()` / `get_active_memory()` around each node   |
| Job wall-clock vs sum-of-stages     | One `eval()` pass on a rebuilt graph vs sum of node durations  |
| Color-coded stage graph             | DOT/SVG colored by `primitive().stream().index`                |

## Findings from discovery (MLX 0.32.4, sources in `../mlx-explore/mlx`)

- Every `array` is a node: `ArrayDesc{ primitive, inputs, siblings }`.
  `depth_first_traversal` from the outputs visits inputs before consumers, so
  the traversal's post-order tape **is** a topological execution order.
- `graph_utils.h` ships `print_graph` / `export_to_dot` with a `NodeNamer`
  (A, B, C…). The built-in DOT lacks shapes, dtypes, streams and bytes, which
  is exactly what a Spark-style UI needs — hence this repo's own exporter.
- Each `Primitive` has a `stream_`, so streams partition the tape into
  stage-like groups and cross-stream edges are visible joins.
- `array::eval()` schedules the node's whole unmaterialized upstream; once
  inputs are materialized it schedules exactly one node. That property is what
  makes per-node attribution possible without touching MLX.
- C++-only APIs that MLX-C (and therefore `../mlx-bridge`) do not expose:
  `print_graph`/`export_to_dot`, per-node traversal, `synchronize(stream)`,
  and `get_peak_memory`-style attribution are C++ surface. A C++ or
  native-library layer is required for any Spark-grade view; see "Caveats".
- Per-node attribution serializes the pipeline (each node waits for the GPU),
  so the honest report shows **two totals**: sum-of-node durations (attributed)
  and a rebuilt single-`eval()` pass (true pipeline wall-clock). Spark has the
  same distinction between stage durations and job duration.

## Caveats, stated honestly

- Timing attribution changes the execution (serializing syncs); it is a
  measurement pass, not a replacement run. The pipeline pass provides the
  uninstrumented number.
- Memory attribution is approximate: the capture holds every intermediate
  alive, so "active" grows with node count, and the Metal allocator may reuse
  buffers. Good enough for visualization, not for accounting.
- No per-kernel GPU timestamps from userspace — for hardware-accurate kernel
  times, `mlx_metal_start_capture/stop_capture` (the `.sysdetail` Instruments
  trace) is the authoritative source. That is a later phase.

## Architecture

One concern per file, `src/`:

| File            | Job                                                                    |
| --------------- | ---------------------------------------------------------------------- |
| `graph_model`   | Walk the graph into nodes/edges/topological order (the captured plan)  |
| `telemetry`     | Time nodes + memory deltas (attributed pass) and pipeline pass         |
| `dot_writer`    | Emit styled Graphviz DOT (stream colors, shapes, bytes on edges)       |
| `svg_renderer`  | Run `dot -Tsvg` (Graphviz) to lay out the DAG                          |
| `report_writer` | Emit the self-contained HTML report (the "Spark UI" for one job)       |
| `hello_world`   | The demo computation: a two-branch model on two streams, then pipeline |

## Build and run

The spike links the **prebuilt** MLX 0.32.4 static library from the
`mlx-bridge` native cache; no MLX build is needed.

```
cmake -S . -B build -DMLX_PREFIX=~/.cache/mlx-bridge/native/dc7fcf8d1fd13ddf/prefix
cmake --build build
./build/hello_world   # writes build/hello_world.html (and .dot/.svg)
open build/hello_world.html
```

Graphviz (`brew install graphviz`) is needed for SVG layout; without it the
report still embeds the raw DOT.

## Roadmap

1. ✅ Hello world: plan capture + per-node timing + stage-colored SVG + report.
2. Compiled vs uncompiled view: `compile()` the same closure and render the
   fused tape next to the eager plan (Spark's "planned vs executed").
3. Rust side: re-express capture via `mlx-c-rust` where the C API allows, so
   the view is reachable from the bridge ecosystem.
4. Metal-capture ingestion: parse the Instruments trace for kernel-accurate
   timings and merge them into the node table.
5. A small live server (Spark-UI-like) instead of a static report.
