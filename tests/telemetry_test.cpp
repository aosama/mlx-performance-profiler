// Integration tests for telemetry's pipeline pass on the Metal GPU. The
// attributed (serial) pass is exercised end to end by the demo; this file
// covers the job-wall measurement added in Phase 1.

#include "telemetry.h"
#include "test_harness.h"

#include "mlx/mlx.h"

#include <vector>

namespace {

using namespace mlx::core;

// --- positive ---

void should_materialize_outputs_and_report_positive_wall() {
  const array job_output = matmul(ones({64, 64}), ones({64, 64}));
  // The pass takes its outputs by value; a copy shares the descriptor, so
  // this handle observes the evaluation.
  const double job_wall_microseconds = mlx_plan::run_pipeline_pass({job_output});
  EXPECT_TRUE(job_wall_microseconds > 0);
  EXPECT_TRUE(job_output.is_available());
}

// --- edge / boundary ---

void should_return_promptly_for_empty_output_list() {
  const double job_wall_microseconds = mlx_plan::run_pipeline_pass({});
  EXPECT_TRUE(job_wall_microseconds >= 0);
}

// --- negative ---

void should_not_fail_when_outputs_are_already_evaluated() {
  array evaluated_output = exp(ones({8}));
  evaluated_output.eval();
  bool has_thrown = false;
  try {
    mlx_plan::run_pipeline_pass({evaluated_output});
  } catch (...) {
    has_thrown = true;
  }
  EXPECT_TRUE(!has_thrown);
  EXPECT_TRUE(evaluated_output.is_available());
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_materialize_outputs_and_report_positive_wall),
      TEST_CASE(should_return_promptly_for_empty_output_list),
      TEST_CASE(should_not_fail_when_outputs_are_already_evaluated),
  });
}
