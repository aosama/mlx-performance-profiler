# Wired memory: how much room the GPU has

On Apple silicon the memory the GPU works on is **wired**: macOS pins it in physical memory. macOS keeps wired memory under a system-wide line, `vm.global_user_wire_limit`. An engine that decides how much to keep in memory (a KV cache, a prompt cache, a second model) needs to know how close it is to that line. The usual estimate is "how much is wired outside MLX": the system's wired total less MLX's active memory and buffer cache. This page shows where that estimate goes wrong, how the profiler shows it for your jobs, and how to read the same figures from your own code.

## What the wired total follows

Measured on one Apple silicon Mac with the probe below and with live reads:

- **The GPU's use wires memory.** Evaluating a 1 GiB array raised the system's wired total by about 1 GiB.
- **Released memory is un-wired a moment later.** MLX stops counting a buffer the instant it releases it, and macOS un-wired the memory about **0.2 s** later (180 to 240 ms across our runs).
- **Memory MLX still holds is un-wired too, once idle.** After about **1.2 s** without GPU use, memory MLX still held was un-wired, and it was wired again the moment the GPU read it.

So "wired outside MLX" errs both ways:

| When | The estimate | Because |
|---|---|---|
| Just after MLX releases memory the GPU used | **too high**, by up to the released amount | MLX no longer counts it; macOS still wires it |
| While memory MLX holds sits idle on the GPU | **too low** | macOS un-wired it, but it counts against the limit again on its next use |

An engine that reads the estimate right after a prompt ends can see several gigabytes of its own memory counted as someone else's for one evaluation, and not on the next. To size safely, count memory MLX holds as wired whether or not macOS currently wires it, and do not trust the estimate for a fraction of a second after a large release.

These figures vary by machine and load. Measure your own Mac's release lag with the probe below.

## See it for your job

Every report now has a **Wired memory** lane, and a procedure comparison has one per procedure:

- **On a report** (`mlx_perf_profile_to_report`, `mlx_perf_compare_to_report`), the samples come from the serial pass: one as it begins, then one after each node's evaluation. That pass keeps every intermediate alive, so MLX's holding only grows. The lane shows how close the job runs to the limit.
- **On a procedure comparison** (`mlx_perf_compare_procedures_to_report`), the samples come from each procedure's observed run, the extra run that is not judged, so sampling never touches the verdict. There is a sample as the run begins and after every blocking wait for the GPU, on any thread: the moments an engine reads its state. There are also samples every millisecond in between and as its GPU work completes. Sampling continues after the run until macOS has un-wired what the run released, for at most 3 s.

The lane draws three lines: the system's wired total, MLX's active memory plus cache, and the part wired outside MLX. The limit is drawn when it is near the data, and evaluation boundaries are tick marks. Each MLX release is a dashed line, labelled with how long macOS took to follow, and samples whose "outside MLX" counted released memory still wired are dots. Under the lane, a table gives:

- the limit;
- the peaks;
- the lowest "outside MLX";
- the least room under the limit;
- whether the run crossed the limit;
- the releases and their lags;
- the samples the lag inflated.

The same figures are in the JSON: `job.wired_memory` in [the measurement file](measurement-file.md#jobwired_memory), and `baseline.wired_memory` and `candidate.wired_memory` in the [procedure comparison record](measurement-file.md#the-procedure-comparison-record-kind-procedure_comparison).

A **release** is a fall of 64 MiB or more in MLX's holding between two samples. Its lag is timed from the first sample that saw MLX's holding fall to the first sample whose wired total had fallen by at least half the release, so a lag is resolved to the sampling interval. A lag that did not end before sampling stopped is `null`, never guessed.

## Read it from your own code, and measure the lag

`mlx_perf_read_wired_memory` takes one reading: the wired total, the limit, MLX's active memory and cache, and the two derived figures. It costs two system calls and MLX's two counters, so it is cheap enough to call after every evaluation. `mlx_perf_measure_wire_release_lag` runs the probe.

1. It waits for the wired total to settle.
2. It allocates and evaluates an array, then releases it and clears MLX's cache.
3. It times from MLX's holding falling to the wired total falling by half the allocation.

It repeats this as many times as you ask and prints progress with an ETA. This complete program does both:

<!-- example: docs/examples/measure_wired_memory.c -->
```c
/*
 * How much room does the GPU have before macOS stops keeping its memory
 * wired, and how long does a release take to show in the system's wired
 * total on this Mac?
 *
 * On Apple silicon the memory the GPU works on is wired, and macOS keeps
 * wired memory under a system-wide limit. This program reads the wired total,
 * the limit and MLX's own holding before and while it holds a 512 MiB array,
 * then measures how long after MLX releases memory macOS un-wires it.
 *
 * Run:    ./measure_wired_memory
 * Writes: nothing; it prints what it read.
 */
#include "mlx/c/array.h"
#include "mlx/c/memory.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"
#include "mlx_perf/mlx_perf_c_api.h"

#include <stdio.h>

static const double MIB = 1024.0 * 1024.0;

static int print_wired_memory(const char* moment) {
  mlx_perf_wired_memory reading;
  if (mlx_perf_read_wired_memory(&reading) != 0) {
    fprintf(stderr, "cannot read wired memory: %s\n", mlx_perf_last_error());
    return 1;
  }
  printf("%-14s wired %8.0f MiB, MLX %6.0f MiB, outside MLX %8.0f MiB, room under the limit %8.0f MiB\n", moment,
         reading.system_wired_bytes / MIB, (reading.mlx_active_bytes + reading.mlx_cache_bytes) / MIB,
         reading.wired_outside_mlx_bytes / MIB, reading.wire_headroom_bytes / MIB);
  return 0;
}

int main(void) {
  mlx_stream stream = mlx_default_gpu_stream_new();
  int status = print_wired_memory("before");

  /* 512 MiB the GPU writes: it is wired while the GPU uses it. */
  const int table_shape[] = {128 * 1024 * 1024};
  mlx_array table = mlx_array_new();
  status |= mlx_ones(&table, table_shape, 1, MLX_FLOAT32, stream);
  status |= mlx_array_eval(table);
  status |= print_wired_memory("while held");
  mlx_array_free(table);
  status |= mlx_synchronize(stream);
  status |= mlx_clear_cache();
  status |= print_wired_memory("just released");

  /* How long macOS keeps released memory wired, on this Mac. */
  mlx_perf_wire_release_lag release_lag;
  if (mlx_perf_measure_wire_release_lag(256ull << 20, 3, &release_lag) != 0) {
    fprintf(stderr, "probe failed: %s\n", mlx_perf_last_error());
    status = 1;
  } else {
    printf("released memory un-wired after %.1f ms (median of %d, largest %.1f ms); limit %.0f MiB\n",
           release_lag.median_release_lag_us / 1e3, release_lag.resolved_count,
           release_lag.largest_release_lag_us / 1e3, release_lag.wire_limit_bytes / MIB);
  }
  mlx_stream_free(stream);
  return status == 0 ? 0 : 1;
}
```

On that Mac it printed:

```text
before         wired     3041 MiB, MLX      0 MiB, outside MLX     3041 MiB, room under the limit    35789 MiB
while held     wired     3608 MiB, MLX    512 MiB, outside MLX     3096 MiB, room under the limit    35222 MiB
just released  wired     3608 MiB, MLX      0 MiB, outside MLX     3608 MiB, room under the limit    35222 MiB
released memory un-wired after 176.1 ms (median of 3, largest 176.3 ms); limit 38830 MiB
```

The third line is the effect at work: MLX's 512 MiB is gone from MLX's count but still wired, so "outside MLX" jumped by 512 MiB.

## Why the wired total comes from sysctl

The profiler reads the wired total from `sysctl vm.page_wired_count`, not `host_statistics64`. macOS rate limits `host_statistics64` for ordinary processes to a few fresh answers a second and returns cached figures after that. A program polling it quickly sees stale totals, and lags measured that way come out near the rate limit's window rather than the real delay.

## Related

- [The measurement file](measurement-file.md#jobwired_memory) for every field.
- [Comparing procedures](comparing-procedures.md), where each procedure's observed run gets a lane.
- [C API reference](reference/c-api.md#mlx_perf_read_wired_memory) for exact signatures.
