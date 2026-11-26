# The measurement file (schema version 1)

Every profile writes its measurements as JSON beside the HTML report:

- `mlx_perf_profile_to_report` writes `<report_file_base>.json` next to the `.html`.
- `mlx_perf_profile_to_measurements` writes only `<measurements_file_base>.json`.
- `hello_world` and `qwen35_profile` write `<base>_<variant>.json` for each variant.
- A comparison (`mlx_perf_compare_to_report`, `hello_world`, `qwen35_profile`)
  also writes `<base>.json`, the comparison record described at the end.

The HTML report and this file are built from the same measurement in the same
call, so they never disagree. Any JSON reader works: Python's `json`, Rust's
`serde_json`, `jq`, a browser.

## Conventions

- **Times** are microseconds (`_us`). Instants such as `issue_started_us`
  count from the start of the timeline pass (`time_origin`); durations count
  from zero.
- **`null` means "not measured", never zero.** A CPU-stream node has no GPU
  window. A node inside a stage-timed run has no timing of its own. A serial
  time is `null` when the serial pass's node at that position ran a different
  operation.
- **Order:** `nodes` are in execution order (`position` 0, 1, 2, …) and
  `stages` in dispatch order. Positions in `input_positions`,
  `previous_on_stream_position` and `node_positions` refer to `nodes`.
- **Members of each object** are written in key order. Readers should look
  fields up by name.
- **`schema_version`** changes when a field is removed or renamed, or changes
  meaning or unit. New fields can appear without a version change, so
  ignore fields you do not know.

## Top level

| Field | Meaning |
|---|---|
| `schema_version` | `1` |
| `mlx_perf_version` | the library version that wrote the file |
| `mlx_version` | the MLX release the job ran on |
| `backend` | `"metal"` |
| `variant_name` | the caller's name for this run |
| `granularity` | `"per_node"`: every node timed alone. `"per_stage"`: stage tasks timed, see `stages` |
| `time_origin` | what instants count from |
| `job` | job totals, below |
| `streams` | one record per MLX stream used |
| `nodes` | one record per executed primitive |
| `stages` | one record per stage task; empty when `per_node` |

## `job`

| Field | Meaning |
|---|---|
| `job_wall_us.median`, `.min`, `.max` | the uninstrumented job duration: one `async_eval` per run, streams overlapping, warm |
| `job_wall_us.samples` | every sample, in the order measured |
| `timeline_wall_us` | the instrumented timeline pass, first issue to last completion |
| `sum_of_stages_us` | the serial pass: each node run alone and synchronised, summed |
| `measured_gpu_execution_us` | the sum of GPU-measured compute |
| `cross_stream_stall_us` | the sum of `wait_on_other_stream_us`; can exceed the wall when streams stall at the same time |
| `executed_node_count`, `gpu_measured_node_count`, `timed_unit_count`, `stream_count` | counts |
| `peak_memory_bytes` | the highest per-node peak during the serial pass, above the memory already allocated when that pass began |
| `command_buffers.timeline_pass_observed` | MLX command buffers seen to complete during the timeline pass |
| `command_buffers.is_lower_bound` | `true` when MLX may have committed buffers the probe never held (stage-timed runs) |
| `graph_build[]` | per pass (`stream_discovery`, `warm_up`, `job_wall`, `timeline`, `serial`): `calls_us`, the time each call spent inside the job's own graph-building code, and `total_us` |

Graph building happens between the passes' clocks: no other number includes
it. `job_wall_us` is evaluation only. Add the `job_wall` calls to it to get
construction plus evaluation.

## `nodes[]`

Identity and structure:

| Field | Meaning |
|---|---|
| `position`, `label` | execution order, and the plan's variable name (as in the report's graph) |
| `op_name` | the primitive's name, e.g. `"Matmul"` |
| `kernel_name` | for a custom Metal kernel (`op_name` `"CustomKernel"`): MLX's name for it, `"custom_kernel_"` plus the author's name. Otherwise `null` |
| `stream_index`, `shape`, `dtype`, `output_bytes` | where it ran and what it produced |
| `input_positions` | every producer it reads |
| `cross_stream_input_positions` | the producers that are on other streams |
| `previous_on_stream_position` | the node before it on its stream, or `null` |
| `stage_position` | its stage task in a `per_stage` run, otherwise `null` |

Timing (`null` in a `per_stage` run, where `stages` holds it):

| Field | Meaning |
|---|---|
| `issue_started_us`, `issue_finished_us` | the host inside `async_eval` for this node |
| `gpu_started_us`, `gpu_finished_us` | the GPU's own execution window, from the node's Metal command buffer |
| `completed_us` | when the host observed the node complete |
| `wait_on_host_us` | host time issuing the node |
| `wait_on_own_stream_us` | earlier work on the same stream still running |
| `wait_on_other_stream_us` | an input from another stream not ready yet |
| `gpu_scheduling_us` | committed, but the GPU had not started it |
| `compute_us` | GPU execution |
| `result_latency_us` | from the GPU finishing to the host noticing |
| `wall_us` | `completed_us − issue_started_us`; equals the six slices above summed |
| `is_compute_gpu_measured` | `true` when `compute_us` comes from the GPU window; `false` for the host-time fallback (CPU streams) |
| `classification` | the largest slice: `compute`, `wait_on_host`, `wait_on_own_stream`, `wait_on_other_stream` or `gpu_scheduling` |
| `gpu_idle_before_us` | this node's GPU start minus the GPU end of the previous node on its stream: how long that GPU queue sat idle in between. `null` for the first node on a stream |

Serial pass, from an identical graph where each node is run alone:

| Field | Meaning |
|---|---|
| `serial_duration_us` | the node synchronised alone |
| `memory_added_bytes` | active memory added by that node |

**Reading `gpu_idle_before_us`.** The timeline pass submits each node
separately so that it can time each one, and the gaps include that
submission cost. Compare gaps with each other, or between two runs of the
same graph, rather than with a plain `eval()`. A plain evaluation batches the
whole graph and has smaller gaps.

## `stages[]` (`per_stage` runs)

Each record has `stage_position`, `stage_index`, `stage_name`,
`stream_index`, `node_positions` and `cross_stream_input_stage_positions`,
plus the same timing fields as a node (including `gpu_idle_before_us` between
consecutive stage tasks on a stream). A stage's GPU window runs from its first
command buffer's start to its last one's end. The README's **Stage
granularity** section explains what that window includes.

## `streams[]`

`stream_index`, `is_gpu`, `node_count` (nodes on the stream, whichever unit
timed them), `first_issue_us`,
`last_completion_us`, `wall_us`, `busy_us` (compute summed) and
`wait_on_other_stream_us`.

## The comparison record (`"kind": "comparison"`)

Written as `<base>.json` beside a comparison page. Per-node detail stays in
each variant's own file, which the record names.

| Field | Meaning |
|---|---|
| `schema_version`, `kind` | `1`, `"comparison"` |
| `baseline`, `candidate` | `variant_name`, and `measurement_file`: that variant's measurement file, relative to this one |
| `metrics.job_wall_us`, `metrics.gpu_span_us`, `metrics.host_submission_us` | one record per metric, below |
| `output_agreement` | `output_count`, and `max_absolute_difference` between the two variants' outputs (float32) |

Each metric record:

| Field | Meaning |
|---|---|
| `baseline`, `candidate` | `median`, `min`, `max`, `lower_quartile`, `upper_quartile`, and every `samples` value in round order |
| `round_difference_us` | the per-round differences (candidate − baseline): `median`, `min`, `max` |
| `difference_interval_us` | `low`, `high` and `coverage` of the distribution-free interval for the median per-round difference (12 rounds: the 3rd and 10th smallest difference, coverage 0.961) |
| `sign_test_p_value` | exact two-sided sign test over the non-zero per-round differences (1 when all are zero) |
| `median_difference_us` | the median per-round difference; negative means the candidate is faster |
| `relative_difference` | `median_difference_us` / the baseline median (0 when that is 0) |
| `verdict` | `candidate_faster` when the whole interval lies below zero, `candidate_slower` when above, otherwise `within_spread` (always, when the interval cannot reach 95% coverage) |
| `rounds_candidate_faster`, `round_count` | rounds, paired by index, in which the candidate's sample was lower |

All three metrics come from 12 rounds measured without per-node
instrumentation. Job wall and host submission (host time inside the job's
single `async_eval`) come from the same runs. GPU span comes from the whole
job timed as one unit per stream: from its first command buffer's GPU start
to its last one's end, summed over streams. It includes GPU gaps while the
host is still encoding, and its end is bounded by an empty marker buffer
(an upper bound by the marker's small scheduling gap, the same for both
variants). The order of the two variants flips every round.

## Example: each node's GPU compute and the gaps between nodes

```python
import json

with open("build/rust_example.json") as measurement_file:
    measurement = json.load(measurement_file)
for node in measurement["nodes"]:
    if not node["is_compute_gpu_measured"]:
        continue  # a CPU-stream node: compute is a host-time fallback
    # compute_us is the GPU's own execution. gpu_finished_us − gpu_started_us
    # is the whole command-buffer window, which for a node joining another
    # stream also includes the wait for that stream's data.
    print(node["position"], node["op_name"], f"{node['compute_us']:.1f} us",
          "idle before:", node["gpu_idle_before_us"])
```
