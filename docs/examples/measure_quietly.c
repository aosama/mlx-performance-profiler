/*
 * Measure a job from inside another program: no console output, no HTML,
 * just the measurement file and the summary, with errors handled.
 *
 * Use this shape in a test, a benchmark harness or a CI job that asserts on
 * numbers. Run:    ./measure_quietly <output directory>
 * Writes:          <output directory>/layer_norm_job.json
 */
#include "mlx/c/array.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"
#include "mlx_perf/mlx_perf_c_api.h"

#include <stdio.h>
#include <string.h>

typedef struct normalization_job {
  mlx_array activations;
  mlx_stream stream;
} normalization_job;

/* (x - mean) / sqrt(var + eps) over the last axis, built fresh per call. */
static int build_normalization(mlx_vector_array fresh_outputs, void* builder_payload) {
  const normalization_job* job = (const normalization_job*)builder_payload;
  mlx_array mean = mlx_array_new(), centered = mlx_array_new(), variance = mlx_array_new();
  mlx_array epsilon = mlx_array_new_float(1e-5f), shifted_variance = mlx_array_new();
  mlx_array deviation = mlx_array_new(), normalized = mlx_array_new();
  int status = mlx_mean_axis(&mean, job->activations, -1, true, job->stream);
  status |= mlx_subtract(&centered, job->activations, mean, job->stream);
  status |= mlx_var_axis(&variance, job->activations, -1, true, 0, job->stream);
  status |= mlx_add(&shifted_variance, variance, epsilon, job->stream);
  status |= mlx_sqrt(&deviation, shifted_variance, job->stream);
  status |= mlx_divide(&normalized, centered, deviation, job->stream);
  status |= mlx_vector_array_append_value(fresh_outputs, normalized);
  mlx_array_free(mean), mlx_array_free(centered), mlx_array_free(variance);
  mlx_array_free(epsilon), mlx_array_free(shifted_variance);
  mlx_array_free(deviation), mlx_array_free(normalized);
  return status;
}

int main(int argument_count, char** arguments) {
  const char* output_directory = argument_count > 1 ? arguments[1] : ".";
  /* The header and the linked library must come from the same release. */
  if (strcmp(mlx_perf_version(), MLX_PERF_VERSION_STRING) != 0) {
    fprintf(stderr, "header %s, library %s\n", MLX_PERF_VERSION_STRING, mlx_perf_version());
    return 1;
  }
  mlx_perf_set_console_output(0); /* the library now prints nothing */

  const int activations_shape[] = {512, 4096};
  normalization_job job = {mlx_array_new(), mlx_default_gpu_stream_new()};
  mlx_ones(&job.activations, activations_shape, 2, MLX_FLOAT32, job.stream);
  mlx_array_eval(job.activations);

  mlx_perf_job_summary summary;
  const int status = mlx_perf_profile_to_measurements(build_normalization, &job, "eager",
                                                      output_directory, "layer_norm_job",
                                                      &summary);
  if (status != 0) {
    /* Read the message before the next mlx_perf_* call on this thread. */
    fprintf(stderr, "measuring failed: %s\n", mlx_perf_last_error());
  } else {
    printf("%s/layer_norm_job.json: %d nodes, job duration %.1f us (%.1f to %.1f)\n",
           output_directory, summary.executed_node_count, summary.job_wall_median_us,
           summary.job_wall_min_us, summary.job_wall_max_us);
  }
  mlx_array_free(job.activations);
  mlx_stream_free(job.stream);
  return status == 0 ? 0 : 1;
}
