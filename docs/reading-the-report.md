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
4. **Data and arithmetic.** This machine's reading and arithmetic ceilings, and for every node (or stage) whether moving its data or doing its arithmetic held it back (below).
5. **Execution graph.** Every operation MLX executed, interactive (below).
6. **How these numbers were measured.** The passes, and what each number does and does not include.

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

## Why is a kernel slow? Its launch shape

The profiler records every kernel launch MLX encodes for each node: the Metal function, the grid and threadgroup sizes, the threadgroups formed, the pipeline's limits and its threadgroup memory. A threadgroup runs on one GPU core, so a launch with fewer threadgroups than the GPU has cores leaves cores idle. That is a common reason a kernel is slow which no timing shows on its own.

- The task table's **Launches** column gives each node's launches, for example `4 threadgroups · 4 of 16 cores`, with ⚠ when some cores have nothing to do.
- In the execution graph, such a node carries a third line, `⚠ 4 of 16 cores`. A stage tile says how many of its launches leave cores idle, so you know which layer to open.
- **Colour: GPU coverage** colours every node from red (few cores busy) to green (every core busy), and grey for nodes that launched no kernel.
- The hover card lists the node's launches: function name, `threadgroups × threads`, and cores with work.

Covering every core rules this reason out but does not prove the launch is efficient: per-core thread and memory limits are not documented, so the profiler does not estimate them. The launches come from the serial pass, where each node runs alone, so they belong to the node exactly, in stage-timed reports too.

## Data movement or arithmetic: what held a kernel back

On Apple silicon the GPU reads unified memory directly. There is no copy step before a kernel: data moves up into the GPU's caches and registers while the kernel runs, overlapped with its arithmetic, as many threadgroups cover each other's waits. So no clock can split a kernel's time between moving data and computing. The **Data and arithmetic** section answers the question that can be answered exactly: how much of its GPU time would each need on its own, at full speed?

- **Ceilings.** Once per process, the profiler measures this machine with MLX's own kernels: how fast the GPU reads unified memory (a reduction over 512 MiB, far larger than any cache), and how fast it multiplies matrices (a 4096 × 4096 × 4096 multiply in `float16` and `bfloat16`, the faster one). The fastest of several runs is kept. They are measured, not taken from a specification, because no kernel reaches an advertised peak.
- **Bytes moved** count every distinct input once at its stored size (a broadcast counts its source, quantized weights count as packed) and every output. A gathered expert multiply (`GatherQMM`, `GatherMM`) counts only the experts its indices named in this run.
- **Arithmetic** is two operations per multiply-add, counted for matrix multiplies and fused attention. Other operations show "not counted" rather than a guess; their arithmetic is small beside their data.
- **Data share** and **arithmetic share** are each need at the ceiling over the unit's GPU-measured compute. The two bars show them side by side.
- **Limited by** names the larger share when it fills at least half the time. **Neither** means more than half the time went to something else: too few threadgroups (see the launch shape above), launch and synchronization costs, or a serial dependency inside the kernel. That is often where a kernel can be made faster, because neither ceiling is in the way.

A share above 100% means the kernel beat the calibration kernel, for example a quantized multiply outrunning the dense one. The ceilings are practical, not physical. Bytes are counted as traffic to unified memory: a small intermediate that stays in a cache between kernels is counted all the same, so for small tensors the data share can read high. Traffic between the caches themselves needs the GPU's hardware counters, which Apple exposes through Instruments, and is not measured here.

In a stage-timed report the section has one row per stage task: its members' bytes and arithmetic summed, judged against the stage's GPU window.

## Stage-timed reports

A large graph (from 256 nodes) whose job names its stages, such as a decoder's layers, is timed one stage at a time: one `async_eval` per stage rather than per node. One Metal command buffer per node would cost several times the job itself. On a Qwen3.5-0.8B (4-bit) decode step at context 1024, per-node timing read 24.19 ms for a 3.15 ms step, while per-stage timing read 3.61 ms.

In a stage-timed report:

- the task table has one row per stage and stream, listing the operations inside;
- the execution graph opens on the stages. A chain of stages is laid out as a snake that fills the screen; click a stage to open its operations, with dashed tiles for the neighbouring stages it reads from and feeds;
- a node inside a stage has no timing of its own. Its cost in the graph is its time when run alone (serial pass), and the hover card shows its stage's breakdown;
- in the JSON, node timing fields are `null` and the `stages` records hold the timing.

## The comparison page

`mlx_perf_compare_to_report` writes a page that opens with the **verdict** (with a note when rounds crossed the wire limit or saw the system compress memory), then the metric table, the memory table (each variant's median and middle half, the per-round difference and its interval), both execution graphs side by side, both timelines on one shared time scale, and the primitive mix: how many of each operation each variant ran, and which operations `mlx::core::compile` fused. See [Comparing two implementations](comparing-implementations.md).

## The wired memory lane

Under the timeline, the **Wired memory** lane plots the system's wired total, MLX's active memory plus cache, and the part wired outside MLX across the serial pass, with the wire limit when it is near the data. The table under it gives the limit, the peaks and the least room under the limit. On Apple silicon the GPU's memory is wired, so that room is what is left before macOS stops keeping more of it wired. See [Wired memory](wired-memory.md).

## The procedure comparison page

`mlx_perf_compare_procedures_to_report` writes a page for two procedures measured whole: the **verdict** on their duration, the metric table for duration, GPU busy and GPU idle time, the **hand-offs per run** (command buffers, blocking waits, host time blocked, GPU busy share), a **wired memory** lane for each procedure's observed run, and their kernel launches by Metal function. There is no execution graph: a procedure is many graphs. See [Comparing procedures](comparing-procedures.md).

## Which number to quote

| Question | Number |
|---|---|
| How long does my job take? | job duration (median, with its range) |
| Is version B faster than A? | the comparison verdict, never two separate job durations |
| How much GPU work is there? | GPU execution |
| Are my streams waiting on each other? | cross-stream stall, and red in the timeline |
| Is the host or the GPU the bottleneck? | the classification mix: mostly host issue means host-bound; mostly queued means the GPU is the bottleneck |
| Which operation is the most expensive? | the hot path, or the cost colours in the graph |
| Is a kernel held back by memory or by arithmetic? | its "Limited by" in Data and arithmetic |

The instrumented duration is always longer than the job duration, because the timeline pass submits nodes one at a time. Do not quote it as the job's duration.
