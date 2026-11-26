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
