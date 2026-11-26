/*
 * Profile one MLX job from C and write its HTML report and measurement file.
 *
 * The job: scores = softmax(x @ x) over the last axis, for a 1024 x 1024
 * input. Build:  see docs/examples/CMakeLists.txt.
 * Run:           ./profile_one_job <output directory>
 * Writes:        <output directory>/softmax_matmul.html and .json
 */
#include "mlx/c/array.h"
#include "mlx/c/ops.h"
#include "mlx/c/random.h"
#include "mlx/c/stream.h"
#include "mlx_perf/mlx_perf_c_api.h"

#include <stdio.h>

typedef struct softmax_job {
  mlx_array input;   /* materialized once, before profiling */
  mlx_stream stream; /* where the job's operations run */
} softmax_job;

/* Called many times by the profiler (11 for one profile). Each call must
 * append the outputs of a FRESH, unevaluated graph: the profiler evaluates
 * them itself, once per pass. Inputs may be reused; operations may not. */
static int build_softmax_job(mlx_vector_array fresh_outputs, void* builder_payload) {
  const softmax_job* job = (const softmax_job*)builder_payload;
  mlx_array products = mlx_array_new();
  mlx_array scores = mlx_array_new();
  int status = mlx_matmul(&products, job->input, job->input, job->stream);
  status |= mlx_softmax_axis(&scores, products, -1, false, job->stream);
  status |= mlx_vector_array_append_value(fresh_outputs, scores);
  mlx_array_free(products);
  mlx_array_free(scores);
  return status; /* non-zero makes the profiler stop and report the failure */
}

int main(int argument_count, char** arguments) {
  const char* output_directory = argument_count > 1 ? arguments[1] : ".";
  const int input_shape[] = {1024, 1024};
  softmax_job job = {mlx_array_new(), mlx_default_gpu_stream_new()};
  mlx_array random_key = mlx_array_new();
  mlx_random_key(&random_key, 7);
  mlx_random_normal(&job.input, input_shape, 2, MLX_FLOAT32, 0.0f, 1.0f, random_key, job.stream);
  mlx_array_eval(job.input); /* inputs hold data; only the job's operations are timed */

  mlx_perf_job_summary summary;
  const int status = mlx_perf_profile_to_report(build_softmax_job, &job, "eager",
                                                "softmax(x @ x), 1024 x 1024",
                                                output_directory, "softmax_matmul", &summary);
  if (status == 0) {
    printf("job duration %.1f us (median), GPU execution %.1f us, %d nodes on %d streams\n",
           summary.job_wall_median_us, summary.measured_gpu_execution_us,
           summary.executed_node_count, summary.stream_count);
  } else {
    fprintf(stderr, "profiling failed: %s\n", mlx_perf_last_error());
  }
  mlx_array_free(random_key);
  mlx_array_free(job.input);
  mlx_stream_free(job.stream);
  return status == 0 ? 0 : 1;
}
