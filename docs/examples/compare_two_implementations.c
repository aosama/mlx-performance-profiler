/*
 * Is my version faster? Compare two implementations of the same job and get
 * a verdict that accounts for run-to-run noise.
 *
 * Baseline:  MLX's softmax over the last axis.
 * Candidate: the same softmax written out: exp(x - max) / sum(exp(x - max)).
 * Both read the same 4096 x 4096 input and must produce outputs of the same
 * count, shape and dtype, or the comparison stops before measuring.
 *
 * Run:    ./compare_two_implementations <output directory>
 * Writes: softmax.html (opens with the verdict), softmax.json (the comparison
 *         as data), and softmax_<variant>.html and .json for each variant.
 */
#include "mlx/c/array.h"
#include "mlx/c/ops.h"
#include "mlx/c/random.h"
#include "mlx/c/stream.h"
#include "mlx_perf/mlx_perf_c_api.h"

#include <stdio.h>

typedef struct softmax_input {
  mlx_array logits;
  mlx_stream stream;
} softmax_input;

static int build_mlx_softmax(mlx_vector_array fresh_outputs, void* builder_payload) {
  const softmax_input* input = (const softmax_input*)builder_payload;
  mlx_array probabilities = mlx_array_new();
  int status = mlx_softmax_axis(&probabilities, input->logits, -1, false, input->stream);
  status |= mlx_vector_array_append_value(fresh_outputs, probabilities);
  mlx_array_free(probabilities);
  return status;
}

static int build_written_out_softmax(mlx_vector_array fresh_outputs, void* builder_payload) {
  const softmax_input* input = (const softmax_input*)builder_payload;
  mlx_array row_max = mlx_array_new(), shifted = mlx_array_new(), exponentials = mlx_array_new();
  mlx_array row_sum = mlx_array_new(), probabilities = mlx_array_new();
  int status = mlx_max_axis(&row_max, input->logits, -1, true, input->stream);
  status |= mlx_subtract(&shifted, input->logits, row_max, input->stream);
  status |= mlx_exp(&exponentials, shifted, input->stream);
  status |= mlx_sum_axis(&row_sum, exponentials, -1, true, input->stream);
  status |= mlx_divide(&probabilities, exponentials, row_sum, input->stream);
  status |= mlx_vector_array_append_value(fresh_outputs, probabilities);
  mlx_array_free(row_max), mlx_array_free(shifted), mlx_array_free(exponentials);
  mlx_array_free(row_sum), mlx_array_free(probabilities);
  return status;
}

static const char* describe_verdict(int verdict) {
  if (verdict == MLX_PERF_VERDICT_CANDIDATE_FASTER) return "candidate faster";
  if (verdict == MLX_PERF_VERDICT_CANDIDATE_SLOWER) return "candidate slower";
  return "within the spread: no result either way";
}

int main(int argument_count, char** arguments) {
  const char* output_directory = argument_count > 1 ? arguments[1] : ".";
  const int logits_shape[] = {4096, 4096};
  softmax_input input = {mlx_array_new(), mlx_default_gpu_stream_new()};
  mlx_array random_key = mlx_array_new();
  mlx_random_key(&random_key, 11);
  mlx_random_normal(&input.logits, logits_shape, 2, MLX_FLOAT32, 0.0f, 4.0f, random_key, input.stream);
  mlx_array_eval(input.logits);

  const mlx_perf_variant baseline = {"mlx_softmax", build_mlx_softmax, &input};
  const mlx_perf_variant candidate = {"written_out", build_written_out_softmax, &input};
  mlx_perf_comparison_summary summary;
  const int status = mlx_perf_compare_to_report(&baseline, &candidate, "Softmax, 4096 x 4096",
                                                output_directory, "softmax", &summary);
  if (status == 0) {
    /* Negative differences mean the candidate took less time. */
    printf("job duration: %s, median difference %+.1f us (%+.1f%%), candidate faster in %d of %d rounds\n",
           describe_verdict(summary.job_wall.verdict), summary.job_wall.median_difference_us,
           100.0 * summary.job_wall.relative_difference, summary.job_wall.rounds_candidate_faster,
           summary.job_wall.round_count);
    printf("largest element-wise output difference: %g\n", summary.max_absolute_output_difference);
  } else {
    fprintf(stderr, "comparison failed: %s\n", mlx_perf_last_error());
  }
  mlx_array_free(random_key);
  mlx_array_free(input.logits);
  mlx_stream_free(input.stream);
  return status == 0 ? 0 : 1;
}
