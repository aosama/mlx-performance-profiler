# Comparing procedures: loops and pipelines, measured whole

A token loop, a sampler, a pipeline with host logic between evaluations: none of these is one MLX graph, so [`mlx_perf_compare_to_report`](comparing-implementations.md) cannot measure them. `mlx_perf_compare_procedures_to_report` can. It runs two pieces of code, each measured whole however many times it evaluates, and says which is faster with the same noise-aware verdict. It also shows where the time went. **GPU busy** is time the GPU was working on the procedure. **GPU idle** is time it sat waiting on the host. Underneath, it counts the **hand-offs** between host and GPU that cause that waiting.

## Compare two procedures

A procedure is a `run` callback and an optional `prepare` callback, both called with your payload. This complete program compares a 32-step loop that reads a value back to the host after every step, as a sampler reads each token, with the same loop evaluated once:

<!-- example: docs/examples/compare_two_procedures.c -->
```c
/*
 * Is my loop faster written another way? Compare two procedures: code that
 * evaluates many MLX graphs, such as a token loop, measured whole.
 *
 * Baseline:  a 32-step loop that reads a value back to the host after every
 *            step, as a sampler reads each token. Each read blocks the host
 *            until the GPU finishes, and the GPU waits for the next step.
 * Candidate: the same 32 steps, evaluated once at the end.
 * Neither is one graph, so mlx_perf_compare_to_report cannot measure them;
 * mlx_perf_compare_procedures_to_report can.
 *
 * Run:    ./compare_two_procedures <output directory>
 * Writes: token_loop.html (the verdict, the hand-offs per run, memory per
 *         run, the kernel launches) and token_loop.json (every round, as data).
 */
#include "mlx/c/array.h"
#include "mlx/c/ops.h"
#include "mlx/c/random.h"
#include "mlx/c/stream.h"
#include "mlx_perf/mlx_perf_c_api.h"

#include <stdbool.h>
#include <stdio.h>

enum { LOOP_STEP_COUNT = 32 };

typedef struct token_loop {
  bool is_reading_every_step;
  mlx_stream stream;
  mlx_array weights;      /* 256 x 256, made once in main */
  mlx_array start_state;  /* 1 x 256, remade by prepare before every run */
  unsigned long long random_seed;
} token_loop;

/* Untimed: a fresh start state before every run, evaluated so its cost
 * never lands in the measured run. */
static int prepare_loop(void* procedure_payload) {
  token_loop* loop = (token_loop*)procedure_payload;
  const int state_shape[] = {1, 256};
  mlx_array random_key = mlx_array_new();
  int status = mlx_random_key(&random_key, loop->random_seed++);
  status |= mlx_random_normal(&loop->start_state, state_shape, 2, MLX_FLOAT32, 0.0f, 1.0f, random_key,
                              loop->stream);
  status |= mlx_array_eval(loop->start_state);
  mlx_array_free(random_key);
  return status;
}

/* Timed: the loop itself, until its last GPU work completes. */
static int run_loop(void* procedure_payload) {
  const token_loop* loop = (const token_loop*)procedure_payload;
  mlx_array state = mlx_array_new(), product = mlx_array_new(), state_sum = mlx_array_new();
  int status = mlx_array_set(&state, loop->start_state);
  for (int step_index = 0; step_index < LOOP_STEP_COUNT && status == 0; ++step_index) {
    status |= mlx_matmul(&product, state, loop->weights, loop->stream);
    status |= mlx_tanh(&state, product, loop->stream);
    if (loop->is_reading_every_step) {
      float sampled_value = 0.0f;
      status |= mlx_sum(&state_sum, state, false, loop->stream);
      status |= mlx_array_item_float32(&sampled_value, state_sum);  /* blocks until the GPU is done */
    }
  }
  status |= mlx_array_eval(state);
  mlx_array_free(state), mlx_array_free(product), mlx_array_free(state_sum);
  return status;
}

static const char* describe_verdict(int verdict) {
  if (verdict == MLX_PERF_VERDICT_CANDIDATE_FASTER) return "candidate faster";
  if (verdict == MLX_PERF_VERDICT_CANDIDATE_SLOWER) return "candidate slower";
  return "within the spread: no result either way";
}

/* Memory verdicts use the same values; lower is better, so "faster" reads
 * as "uses less". */
static const char* describe_memory_verdict(int verdict) {
  if (verdict == MLX_PERF_VERDICT_CANDIDATE_FASTER) return "candidate uses less";
  if (verdict == MLX_PERF_VERDICT_CANDIDATE_SLOWER) return "candidate uses more";
  return "within the spread: no result either way";
}

int main(int argument_count, char** arguments) {
  const char* output_directory = argument_count > 1 ? arguments[1] : ".";
  const int weights_shape[] = {256, 256};
  mlx_stream stream = mlx_default_gpu_stream_new();
  mlx_array weights = mlx_array_new(), random_key = mlx_array_new();
  mlx_random_key(&random_key, 7);
  mlx_random_normal(&weights, weights_shape, 2, MLX_FLOAT32, 0.0f, 0.05f, random_key, stream);
  mlx_array_eval(weights);

  token_loop reading_loop = {true, stream, weights, mlx_array_new(), 1};
  token_loop batched_loop = {false, stream, weights, mlx_array_new(), 1};
  const mlx_perf_procedure baseline = {"read_every_step", prepare_loop, run_loop, &reading_loop};
  const mlx_perf_procedure candidate = {"eval_once", prepare_loop, run_loop, &batched_loop};
  mlx_perf_procedure_comparison_summary summary;
  const int status = mlx_perf_compare_procedures_to_report(&baseline, &candidate, "A 32-step token loop",
                                                           output_directory, "token_loop", &summary);
  if (status == 0) {
    printf("duration: %s, median difference %+.1f us (%+.1f%%)\n", describe_verdict(summary.duration.verdict),
           summary.duration.median_difference_us, 100.0 * summary.duration.relative_difference);
    printf("GPU idle: %.1f us vs %.1f us per run; blocking waits: %.0f vs %.0f per run\n",
           summary.gpu_idle.baseline_median_us, summary.gpu_idle.candidate_median_us,
           summary.baseline.median_blocking_wait_count, summary.candidate.median_blocking_wait_count);
    /* Memory, judged on the same rounds: the run's own peak MLX working set. */
    printf("peak MLX working set: %s, %.1f KiB vs %.1f KiB; rounds over the wire limit: %d vs %d\n",
           describe_memory_verdict(summary.peak_mlx_active_added.verdict),
           summary.peak_mlx_active_added.baseline_median_bytes / 1024.0,
           summary.peak_mlx_active_added.candidate_median_bytes / 1024.0,
           summary.baseline_rounds_crossing_wire_limit, summary.candidate_rounds_crossing_wire_limit);
  } else {
    fprintf(stderr, "comparison failed: %s\n", mlx_perf_last_error());
  }
  mlx_array_free(reading_loop.start_state), mlx_array_free(batched_loop.start_state);
  mlx_array_free(weights), mlx_array_free(random_key);
  mlx_stream_free(stream);
  return status == 0 ? 0 : 1;
}
```

On an Apple-silicon Mac it prints, for example:

```text
duration: candidate faster, median difference -5559.8 us (-89.6%)
GPU idle: 4737.0 us vs 219.0 us per run; blocking waits: 32 vs 1 per run
peak MLX working set: candidate uses more, 5.0 KiB vs 33.0 KiB; rounds over the wire limit: 0 vs 0
```

The last line is the price of the speed-up: evaluating once keeps every step's intermediate alive until the end, so its peak working set is larger, small here and large for a real model.

Both loops do the same 32 matrix products. The one that reads back every step takes about 6.2 ms, of which the GPU is idle for 4.7 ms: each read stops the host until the GPU finishes, then the GPU waits while the host builds and submits the next step. The table of hand-offs on the page, `token_loop.html`, shows the cause directly: 32 blocking waits and 32 command buffers per run, against 1.

## The procedure contract

- **`run` is measured whole**: from the moment it starts until every command buffer it committed has completed on the GPU. Work it leaves in flight (an `async_eval` it never waits for) is still counted, because the measurement waits for it.
- **`prepare` runs before every run, untimed.** Put setup there: fresh inputs, a reset KV cache, a new random seed. Evaluate what it creates before returning (`mlx_array_eval`), so that work is not left to the timed run. It may be `NULL`.
- **Both are called on the calling thread, but the work may run anywhere.** An engine with an owner thread for MLX can have `run` send a request and wait for the reply. Everything the process submits to the GPU while `run` runs (command buffers, blocking waits, kernel launches) is counted as the procedure's, on whichever thread it happens, so use MLX for nothing else while a comparison runs.
- **Return `0` from each.** Any other value stops the comparison, writing nothing, with `procedure <name>: run returned <n>` (or `prepare returned <n>`) in `mlx_perf_last_error()`.
- **The procedure may evaluate as often as it likes**: `mlx_array_eval`, `mlx_eval`, reading an item, `mlx_synchronize`. Each blocking one is counted.

Each procedure runs 14 times: one warm-up, one observed run that records its kernel launches, and one run in each of the 12 rounds.

## How the verdict is decided

Exactly as for [two graph jobs](comparing-implementations.md#how-the-verdict-is-decided): 12 rounds, each running both procedures, flipping which goes first; the verdict is judged on the per-round differences, and a difference counts only when the whole 96% interval lies on one side of zero (at least 10 of 12 rounds). Three time metrics are judged, each an `mlx_perf_metric_comparison` in `mlx_perf_procedure_comparison_summary` (three memory metrics are judged on the same rounds too: see [Memory, judged on the same rounds](#memory-judged-on-the-same-rounds)):

| Metric | What it measures |
|---|---|
| `duration` | the run, until its GPU work completed: the headline |
| `gpu_busy` | time inside the run when at least one of its command buffers was on the GPU (the union of their GPU windows, from Metal's own timestamps) |
| `gpu_idle` | `duration` minus `gpu_busy`: the GPU had nothing of the procedure's to do, because the host was building, submitting or waiting |

When `duration` improves and most of the improvement is in `gpu_idle`, the faster procedure did not make the GPU work less: it kept the GPU fed. When it is in `gpu_busy`, the kernels themselves got faster or fewer.

## Reading the hand-offs

The page's **Hand-offs per run** table, and `baseline` and `candidate` in the summary, give each procedure's medians over the rounds:

| Column (summary field) | Meaning |
|---|---|
| Command buffers (`median_command_buffer_count`) | MLX's submissions to the GPU. MLX starts a new one at each evaluation, and when a buffer grows past its size limit |
| Blocking waits (`median_blocking_wait_count`) | evaluations and synchronizes that stopped the host until the GPU finished. Each is a point where neither side can run ahead |
| Host time blocked (`median_blocking_wait_us`) | the host time spent in those waits |
| GPU busy of the run | `gpu_busy` over `duration`, as a share |
| Kernel launches (`kernel_launch_count`) | from the observed run; `launches_unable_to_fill_gpu` counts those with fewer threadgroups than GPU cores (`-1` when the core count is unknown) |

The number of blocking waits is often the finding. A loop that must read each step's result (sampling a token) cannot drop to one, but it can overlap: start the next step with `async_eval` before reading the current one.

Under the table, the page lists the kernel launches of both procedures side by side, grouped by Metal function, with ⚠ where a launch leaves GPU cores without work, as on [the graph comparison page](comparing-implementations.md#why-the-slower-one-is-slower-kernel-launches-side-by-side).

## Memory, judged on the same rounds

Under the hand-offs, **Memory per run** judges three memory figures exactly as the times are judged, on the same 12 rounds. Each is read just before and just after every run, never during it, so the timing pays nothing for it:

| Metric (summary field) | What it measures |
|---|---|
| Peak MLX working set (`peak_mlx_active_added`) | MLX's peak active memory during the run, less its active memory as the run began. MLX's counters are process-wide, so the raw peak would include memory the other procedure keeps alive between its runs |
| Left behind (`mlx_held_change`) | MLX's active memory plus cache at the end, less at the start: what one run keeps, such as a cache that is never trimmed |
| System wired at the run's end (`system_wired_at_end`) | the system's wired total as the run ended. Its true peak needs sampling inside the run, which the judged rounds never do; the [wired memory](wired-memory.md) lane has it for the observed run |

The verdict reads "less" or "more" instead of "faster" or "slower", and the table's last column counts the rounds the candidate used less. A change that trades memory for time shows both sides on one page.

**When a round crossed the line, the page says so beside the timing verdict.** If a run crossed the wire limit, or the system compressed memory during it, a note under the verdict counts those rounds for each procedure. A run crossed the limit when the kernel refused a wire request over it (`vm.add_wire_count_over_global_limit` rose) or the wired total ended over it. Past the limit, macOS stops keeping part of the GPU's memory wired, compresses it and may swap it, and the run slows for a reason outside the code. A timing verdict built on such rounds is not the code's doing. Compression is counted system-wide, so another program's memory pressure shows too; the note says so. The summary carries the same counts (`baseline_rounds_crossing_wire_limit` and the others), and every round's figures are in the JSON.

## When to use which comparison

| Your two implementations are… | Use |
|---|---|
| one graph each, the same outputs | `mlx_perf_compare_to_report`: per-node timing, both execution graphs, timelines and primitive mix |
| more than one graph, or graphs with host code between them | `mlx_perf_compare_procedures_to_report`: duration, GPU busy and idle, hand-offs and launches |

A procedure comparison does not see inside the run node by node; it sees the run as the GPU and the host experienced it. To find the slow node of one step, profile that step as a graph job ([Profiling a job](profiling-a-job.md)).

## What is written

| File | Contents |
|---|---|
| `<base>.html` | the verdict and its metric table, the hand-offs per run, memory per run, a [wired memory](wired-memory.md) lane for each procedure's observed run, the kernel launches by Metal function |
| `<base>.json` | the procedure comparison record: every round of both procedures, the verdicts and the launches ([format](measurement-file.md#the-procedure-comparison-record-kind-procedure_comparison)) |

Procedure names follow the rules of [variant names](comparing-implementations.md#naming-variants): non-empty, no `/`, distinct ignoring case.

## Related

- [C API reference](reference/c-api.md#mlx_perf_compare_procedures_to_report) for exact signatures.
- [Troubleshooting](troubleshooting.md#a-procedure-step-reported-a-failure) for its error messages.
