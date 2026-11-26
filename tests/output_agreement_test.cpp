// Integration tests for output_agreement on real (evaluated) MLX arrays.

#include "output_agreement.h"
#include "test_harness.h"

#include "mlx/mlx.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace mlx::core;
using mlx_perf::compare_outputs;

std::string invalid_argument_message(const std::vector<array>& baseline_outputs,
                                     const std::vector<array>& candidate_outputs) {
  try {
    compare_outputs(baseline_outputs, candidate_outputs);
  } catch (const std::invalid_argument& mismatch) {
    return mismatch.what();
  }
  return "";
}

// --- positive ---

void should_measure_the_largest_difference_between_matching_outputs() {
  const array baseline_output = array({1.0f, 2.0f, 3.0f});
  const array candidate_output = array({1.0f, 2.5f, 2.75f});
  const auto output_agreement = compare_outputs({baseline_output}, {candidate_output});
  EXPECT_TRUE(output_agreement.output_count == 1);
  EXPECT_TRUE(output_agreement.max_absolute_difference == 0.5);
}

void should_find_identical_outputs_in_full_agreement() {
  const array product = matmul(ones({8, 8}), ones({8, 8}));
  const auto output_agreement = compare_outputs({product, exp(product)}, {product, exp(product)});
  EXPECT_TRUE(output_agreement.output_count == 2);
  EXPECT_TRUE(output_agreement.max_absolute_difference == 0.0);
}

// --- edge / boundary ---

void should_agree_on_empty_and_half_precision_outputs() {
  EXPECT_TRUE(compare_outputs({zeros({0})}, {zeros({0})}).max_absolute_difference == 0.0);
  const auto half_agreement = compare_outputs({array({1.0f}, float16)}, {array({1.5f}, float16)});
  EXPECT_TRUE(half_agreement.max_absolute_difference == 0.5);
  EXPECT_TRUE(compare_outputs({}, {}).output_count == 0);
}

// --- negative ---

void should_refuse_a_different_number_of_outputs() {
  EXPECT_TRUE(invalid_argument_message({ones({2})}, {ones({2}), ones({2})}).find("2") !=
              std::string::npos);
}

void should_refuse_a_different_shape_naming_both() {
  const std::string mismatch = invalid_argument_message({ones({2, 3})}, {ones({3, 2})});
  EXPECT_TRUE(mismatch.find("(2,3)") != std::string::npos);
  EXPECT_TRUE(mismatch.find("(3,2)") != std::string::npos);
}

void should_refuse_a_different_dtype() {
  const std::string mismatch =
      invalid_argument_message({ones({4}, float32)}, {ones({4}, float16)});
  EXPECT_TRUE(mismatch.find("float16") != std::string::npos);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_measure_the_largest_difference_between_matching_outputs),
      TEST_CASE(should_find_identical_outputs_in_full_agreement),
      TEST_CASE(should_agree_on_empty_and_half_precision_outputs),
      TEST_CASE(should_refuse_a_different_number_of_outputs),
      TEST_CASE(should_refuse_a_different_shape_naming_both),
      TEST_CASE(should_refuse_a_different_dtype),
  });
}
