# Comparing two implementations

Is your custom Metal kernel faster than MLX's own? Did `mlx::core::compile` help? Is the rewrite worth it? `mlx_perf_compare_to_report` answers with a verdict that accounts for run-to-run noise, instead of two numbers you have to eyeball.

## Compare two versions of a job

Give it two variants, each a name and a callback that builds the job its own way. The two must produce the same number of outputs, with the same shapes and dtypes. This complete program compares MLX's softmax with a written-out softmax on a 4096 × 4096 input:

<!-- example: docs/examples/compare_two_implementations.c -->
```c
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
```

On an Apple-silicon Mac it prints, for example:

```text
job duration: candidate slower, median difference +1455.5 us (+111.8%), candidate faster in 0 of 12 rounds
largest element-wise output difference: 1.78814e-07
```

The written-out version runs seven operations (max, two broadcasts, subtract, exp, sum, divide) where MLX's softmax runs one, so it loses every round. The report, `softmax.html`, opens with the same verdict, then both execution graphs side by side, both timelines on one time scale, and the primitive mix.

## How the verdict is decided

Measuring the two jobs one after the other is biased: whatever drifts on the machine (clocks, temperature, other work) lands on one of them only. In one measurement of a Qwen3.5 decode step, the variant measured second read 4.2 ms against 3.2 ms, although interleaved runs show it faster every time. So the comparison:

1. checks the two jobs' outputs agree in count, shape and dtype, and records their largest element-wise difference;
2. runs **12 rounds**, each running both variants once, uninstrumented, **flipping which goes first** every round;
3. computes each round's difference, candidate minus baseline, so drift the two share within a round cancels;
4. takes the **median difference** and a distribution-free interval for it: with 12 rounds, the 3rd to the 10th smallest difference, which covers the true median 96.1% of the time;
5. calls the candidate **faster** only if the whole interval is below zero (in practice, it won at least 10 of 12 rounds), **slower** if the whole interval is above zero, and otherwise **within the spread**.

Two identical variants are called different less than 4% of the time per metric. A test asserts that bound by simulation.

## Reading the result

`mlx_perf_comparison_summary` holds three metrics, each an `mlx_perf_metric_comparison`:

| Metric | What it measures |
|---|---|
| `job_wall` | the job's duration, one `async_eval`, streams overlapping: the headline |
| `gpu_span` | from the job's first command buffer starting on the GPU to its last one ending, summed over streams |
| `host_submission` | host time inside the job's `async_eval`: encoding and committing the work |

For each, `verdict` is `MLX_PERF_VERDICT_CANDIDATE_FASTER` (-1), `MLX_PERF_VERDICT_WITHIN_SPREAD` (0) or `MLX_PERF_VERDICT_CANDIDATE_SLOWER` (1). `median_difference_us` is negative when the candidate took less time. `relative_difference` is that difference over the baseline's median. `rounds_candidate_faster` of `round_count` says how often the candidate won. `difference_interval_low_us` to `difference_interval_high_us` is the interval the verdict was judged on.

**Within the spread is not a result.** It means the difference did not exceed run-to-run noise. Report it as "no difference found at this size", never as slightly faster or slightly slower. On a busy machine or a sub-millisecond job, a real but small difference can land here. Measure on a quiet machine, or make the job larger.

**The verdict judges time, not correctness.** `max_absolute_output_difference` (in the summary and in the JSON's `output_agreement`) is reported but not judged. Check it against your own tolerance. It is `NaN` in C (`null` in the JSON) when any difference is not finite.

## Comparing a custom Metal kernel

Build the candidate with MLX-C's `mlx_fast_metal_kernel_*` functions (or `mlx::core::fast::metal_kernel` in C++) and the baseline with the stock operation. Every custom kernel's primitive is named `CustomKernel`, so the measurement file adds `kernel_name`: MLX's name for that kernel, `custom_kernel_` followed by the name you gave it. Two custom kernels in one job stay distinguishable in the report, the graph and the JSON.

## Naming variants

Variant names become part of file names: `<report_file_base>_<variant_name>.html` and `.json`. A name must be non-empty, contain no `/`, and differ from the other variant's name ignoring case, because on macOS's default file system `Stock` and `stock` name the same file. A name that breaks these rules fails before anything is measured. See [Troubleshooting](troubleshooting.md#both-variants-are-named-the-same).

## What is written

| File | Contents |
|---|---|
| `<base>.html` | the comparison page: verdict, metrics, both execution graphs, shared-scale timelines, primitive mix |
| `<base>.json` | the comparison record: every round's numbers and the verdicts ([format](measurement-file.md#the-comparison-record-kind-comparison)) |
| `<base>_<variant>.html`, `<base>_<variant>.json` | the full report and measurement file of each variant |

A comparison builds each variant's graph 28 times through the C API. If the outputs differ in count, shape or dtype, it fails before measuring and writes nothing.

## Related

- [The measurement file](measurement-file.md) to read the per-round numbers yourself.
- [C API reference](reference/c-api.md#mlx_perf_compare_to_report) for exact signatures.
