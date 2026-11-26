# Reading the report

What each part of the HTML report tells you, and how to use the interactive execution graph. A report is one self-contained file: open it in any browser, offline. It follows the system's light or dark theme.

## The six slices: where a node's time goes

Every timed node's wall time, from the host starting to issue it to the host seeing it complete, is split into six slices that always add up to the wall. They follow the questions Apache Spark's StagePage answers:

| Slice | Meaning | Spark's equivalent |
|---|---|---|
| host issue | host time inside `async_eval`, encoding the node | Scheduler Delay (host side) |
| queued | earlier work on the same stream was still running | (none: stream queueing) |
| stalled | an input from another stream was not ready yet | Shuffle Read `fetchWaitTime` |
| GPU scheduling | committed, but the GPU had not started it | Scheduler Delay |
| compute | GPU execution, from the node's Metal command buffer | Executor Computing Time |
| result | from the GPU finishing to the host noticing | Getting Result Time |

A node's **classification** is its largest slice other than result: `compute`, `wait_on_host`, `wait_on_own_stream` (queued), `wait_on_other_stream` (stalled) or `gpu_scheduling`. Compute is measured by the GPU itself (`GPUStartTime` and `GPUEndTime` of the node's command buffer), not inferred. A `~` marks the rare fallback to host time, for nodes on a CPU stream.

## The report, top to bottom

1. **Job summary.** Cards for job duration (the honest end-to-end number, uninstrumented), GPU execution, cross-stream stall, instrumented duration, sum of stages (every node run alone) and peak memory. Under them, **where node time went** is one bar of the six slices summed over all nodes.
2. **Timeline.** One lane for the host thread that issues nodes, and one lane per MLX stream. On a stream lane, red is stalled, amber is GPU scheduling, green is GPU compute, and blank means nothing was runnable on that stream. An arrow runs from a producer's GPU end to where its consumer on another stream stopped waiting. Under it, a table per stream gives wall, busy, stalled and utilization.
3. **Tasks.** One sortable row per node: every slice, the wall, the node run alone ("Serial"), its class and a stacked bar.
4. **Execution graph.** Every operation MLX executed, interactive (below).
5. **How these numbers were measured.** The passes, and what each number does and does not include.

## Use the execution graph

- **Colour** is where the node's time went, in the timeline's colours. **Brightness** is cost: the most expensive nodes are the most vivid.
- **Hover** a node for its op, kernel name, shape, dtype, output size, stream, stage and six-slice bar.
- **Click** a node to keep everything it depends on and everything that depends on it, and dim the rest. Click the background to clear.
- **Search** (press `/`) by op name, kernel name or label; Enter steps through the matches.
- **Hot path** highlights the most expensive chain of dependent nodes, end to end, and its total cost.
- **Colour** menu: by where time went, by cost (a heat scale) or by stream.
- **Fit** (or `f`) shows the whole graph; **⤢** goes full screen; **PNG** downloads the whole graph as an image.
- A **dashed red edge** is a read from another stream: the join where a stall happens.

The graph reads the same measurement document as the `.json` file, so the picture and the data cannot disagree. A page script can drive it, as the release images are captured: `mlxPerfGraphView.mountedViews[0]` has `openStage(position)`, `showOverview("stages" | "nodes")`, `setColourMode("time" | "heat" | "stream")`, `showHotPath()` and `fit()`.

## Stage-timed reports

A large graph (from 256 nodes) whose job names its stages, such as a decoder's layers, is timed one stage at a time: one `async_eval` per stage rather than per node. One Metal command buffer per node would cost several times the job itself. On a Qwen3.5-0.8B (4-bit) decode step at context 1024, per-node timing read 24.19 ms for a 3.15 ms step, while per-stage timing read 3.61 ms.

In a stage-timed report:

- the task table has one row per stage and stream, listing the operations inside;
- the execution graph opens on the stages. A chain of stages is laid out as a snake that fills the screen; click a stage to open its operations, with dashed tiles for the neighbouring stages it reads from and feeds;
- a node inside a stage has no timing of its own. Its cost in the graph is its time when run alone (serial pass), and the hover card shows its stage's breakdown;
- in the JSON, node timing fields are `null` and the `stages` records hold the timing.

## The comparison page

`mlx_perf_compare_to_report` writes a page that opens with the **verdict**, then the metric table (each variant's median and middle half, the per-round difference and its interval), both execution graphs side by side, both timelines on one shared time scale, and the primitive mix: how many of each operation each variant ran, and which operations `mlx::core::compile` fused. See [Comparing two implementations](comparing-implementations.md).

## Which number to quote

| Question | Number |
|---|---|
| How long does my job take? | job duration (median, with its range) |
| Is version B faster than A? | the comparison verdict, never two separate job durations |
| How much GPU work is there? | GPU execution |
| Are my streams waiting on each other? | cross-stream stall, and red in the timeline |
| Is the host or the GPU the bottleneck? | the classification mix: mostly host issue means host-bound; mostly queued means the GPU is the bottleneck |
| Which operation is the most expensive? | the hot path, or the cost colours in the graph |

The instrumented duration is always longer than the job duration, because the timeline pass submits nodes one at a time. Do not quote it as the job's duration.
