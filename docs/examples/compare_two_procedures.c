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
