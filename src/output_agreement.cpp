#include "output_agreement.h"

#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "mlx/utils.h"

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace mlx_perf {
namespace {

std::string describe_output(const mlx::core::array& output_array) {
  std::ostringstream output_description;
  output_description << output_array.shape() << " " << output_array.dtype();
  return output_description.str();
}

void require_same_kind(const mlx::core::array& baseline_output,
                       const mlx::core::array& candidate_output, std::size_t output_index) {
  if (baseline_output.shape() != candidate_output.shape() ||
      baseline_output.dtype() != candidate_output.dtype()) {
    throw std::invalid_argument(
        "output " + std::to_string(output_index) + " differs: baseline " +
        describe_output(baseline_output) + ", candidate " + describe_output(candidate_output));
  }
}

// float32 for every dtype: half-precision differences would otherwise round
// to the very precision being compared, and integers need no special case.
// NaN when any difference is not finite: the GPU max reduction does not
// reliably propagate NaN, so finiteness is checked on its own.
double max_absolute_difference(const mlx::core::array& baseline_output,
                               const mlx::core::array& candidate_output) {
  if (baseline_output.size() == 0) {
    return 0.0;
  }
  const mlx::core::array element_differences = mlx::core::abs(
      mlx::core::subtract(mlx::core::astype(baseline_output, mlx::core::float32),
                          mlx::core::astype(candidate_output, mlx::core::float32)));
  mlx::core::array are_all_finite = mlx::core::all(mlx::core::isfinite(element_differences));
  mlx::core::array largest_difference = mlx::core::max(element_differences);
  mlx::core::eval({are_all_finite, largest_difference});
  if (!are_all_finite.item<bool>()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return static_cast<double>(largest_difference.item<float>());
}

} // namespace

OutputAgreement compare_outputs(const std::vector<mlx::core::array>& baseline_outputs,
                                const std::vector<mlx::core::array>& candidate_outputs) {
  if (baseline_outputs.size() != candidate_outputs.size()) {
    throw std::invalid_argument("the baseline produces " +
                                std::to_string(baseline_outputs.size()) +
                                " outputs, the candidate " +
                                std::to_string(candidate_outputs.size()));
  }
  OutputAgreement output_agreement;
  output_agreement.output_count = static_cast<int>(baseline_outputs.size());
  for (std::size_t output_index = 0; output_index < baseline_outputs.size(); ++output_index) {
    require_same_kind(baseline_outputs[output_index], candidate_outputs[output_index],
                      output_index);
    const double output_difference =
        max_absolute_difference(baseline_outputs[output_index], candidate_outputs[output_index]);
    // std::max(0, NaN) is 0: a non-finite difference must win explicitly.
    if (std::isnan(output_difference) ||
        output_difference > output_agreement.max_absolute_difference) {
      output_agreement.max_absolute_difference = output_difference;
    }
    if (std::isnan(output_agreement.max_absolute_difference)) {
      break;
    }
  }
  return output_agreement;
}

} // namespace mlx_perf
