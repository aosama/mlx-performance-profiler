#pragma once

// Output agreement: one concern only — check that two graphs compared as
// implementations of one job produce outputs of the same count, shapes and
// dtypes, and measure how far their values differ.

#include "mlx/array.h"

#include <vector>

namespace mlx_perf {

struct OutputAgreement {
  int output_count = 0;
  // Largest |baseline − candidate| over every element of every output, in
  // float32. Reported, never enforced: a custom kernel and a stock
  // primitive may legitimately differ in the last bits. NaN when any
  // element's difference is not finite (a NaN or infinity in either output,
  // or infinities on both sides), so a broken kernel can never read as
  // perfect agreement. Limits of the float32 view: differences below
  // float32 resolution in int64/float64 outputs, and the imaginary part of
  // complex outputs, are not seen.
  double max_absolute_difference = 0.0;
};

// Both sets must already be evaluated (they are after a warm-up). Throws
// std::invalid_argument, naming both sides, when the count, a shape or a
// dtype differs: timing two graphs that compute different things answers no
// question.
OutputAgreement compare_outputs(const std::vector<mlx::core::array>& baseline_outputs,
                                const std::vector<mlx::core::array>& candidate_outputs);

} // namespace mlx_perf
