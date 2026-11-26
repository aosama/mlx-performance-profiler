/*
 * A consumer of the INSTALLED mlx_perf package, written in plain C: it sees
 * only <mlx_perf/mlx_perf_c_api.h>, MLX-C, and what find_package(mlx_perf)
 * provides. It profiles a two-stream job and checks the summary.
 *
 * Usage: mlx_perf_consumer <output directory>
 */
#include "mlx/c/array.h"
#include "mlx/c/device.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"
#include "mlx_perf/mlx_perf_c_api.h"

#include <stdio.h>
#include <string.h>

typedef struct consumer_streams {
  mlx_stream side_stream;
  mlx_stream main_stream;
} consumer_streams;

/* matmul on a side stream, exp on the main stream: one cross-stream join. */
static int build_consumer_graph(mlx_vector_array fresh_outputs, void* builder_payload) {
  const consumer_streams* streams = (const consumer_streams*)builder_payload;
  const int shape[] = {64, 64};
  mlx_array square_ones = mlx_array_new();
  mlx_array product = mlx_array_new();
  mlx_array activated = mlx_array_new();
  int status = mlx_ones(&square_ones, shape, 2, MLX_FLOAT32, streams->side_stream);
  status |= mlx_matmul(&product, square_ones, square_ones, streams->side_stream);
  status |= mlx_exp(&activated, product, streams->main_stream);
  status |= mlx_vector_array_append_value(fresh_outputs, activated);
  mlx_array_free(square_ones);
  mlx_array_free(product);
  mlx_array_free(activated);
  return status;
}

int main(int argument_count, char** arguments) {
  if (argument_count != 2) {
    fprintf(stderr, "usage: %s <output directory>\n", arguments[0]);
    return 2;
  }
  /* A header and an archive from different releases must not pass silently. */
  if (strcmp(mlx_perf_version(), MLX_PERF_VERSION_STRING) != 0) {
    fprintf(stderr, "header %s but library %s\n", MLX_PERF_VERSION_STRING,
            mlx_perf_version());
    return 1;
  }
  consumer_streams streams = {mlx_stream_new_device(mlx_device_new_type(MLX_GPU, 0)),
                              mlx_default_gpu_stream_new()};
  mlx_perf_job_summary summary;
  memset(&summary, 0, sizeof(summary));
  const int status = mlx_perf_profile_to_report(
      build_consumer_graph, &streams, "consumer", "Installed-package consumer",
      arguments[1], "consumer_report", &summary);
  mlx_stream_free(streams.side_stream);
  mlx_stream_free(streams.main_stream);
  if (status != 0) {
    fprintf(stderr, "profile failed: %s\n", mlx_perf_last_error());
    return 1;
  }
  printf("mlx_perf %s: %d nodes on %d streams, job wall %.1f us\n", mlx_perf_version(),
         summary.executed_node_count, summary.stream_count, summary.job_wall_median_us);
  return summary.executed_node_count > 0 && summary.stream_count == 2 ? 0 : 1;
}
