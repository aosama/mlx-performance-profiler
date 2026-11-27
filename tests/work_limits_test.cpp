// Unit tests for work_limits: pricing a unit's bytes and
// arithmetic at the ceilings, and naming what held it back. Synthetic
// numbers, no GPU.

#include "test_harness.h"
#include "work_limits.h"

namespace {

using namespace mlx_perf;

// Round numbers: 1,000 bytes per microsecond reading, 1,000,000 operations
// per microsecond of arithmetic.
MachineCeilings round_ceilings() {
  return {true, 1e9, 1e12, "bfloat16"};
}

NodeWork work_of(std::uint64_t bytes_read, std::uint64_t bytes_written, double operations) {
  NodeWork unit_work;
  unit_work.bytes_read = bytes_read;
  unit_work.bytes_written = bytes_written;
  unit_work.is_arithmetic_counted = operations > 0.0;
  unit_work.arithmetic_operations = operations;
  return unit_work;
}

// --- positive ---

void should_name_data_movement_when_the_bytes_fill_most_of_the_time() {
  const WorkJudgement judgement = judge_work(work_of(80'000, 20'000, 0.0), round_ceilings(), 125.0, true);
  EXPECT_NEAR(judgement.data_movement_microseconds, 100.0, 1e-9);
  EXPECT_NEAR(judgement.data_movement_share, 0.8, 1e-9);
  EXPECT_TRUE(!judgement.has_arithmetic_time);
  EXPECT_TRUE(judgement.limited_by == WorkLimit::data_movement);
}

void should_name_arithmetic_when_the_operations_fill_most_of_the_time() {
  const WorkJudgement judgement = judge_work(work_of(10'000, 0, 90e6), round_ceilings(), 100.0, true);
  EXPECT_NEAR(judgement.arithmetic_microseconds, 90.0, 1e-9);
  EXPECT_NEAR(judgement.arithmetic_share, 0.9, 1e-9);
  EXPECT_NEAR(judgement.data_movement_share, 0.1, 1e-9);
  EXPECT_TRUE(judgement.limited_by == WorkLimit::arithmetic);
}

void should_name_neither_when_both_needs_are_small() {
  const WorkJudgement judgement = judge_work(work_of(20'000, 0, 30e6), round_ceilings(), 100.0, true);
  EXPECT_TRUE(judgement.limited_by == WorkLimit::neither);
}

// --- edge / boundary ---

void should_count_a_share_of_exactly_one_half_as_the_limit() {
  const WorkJudgement judgement = judge_work(work_of(50'000, 0, 0.0), round_ceilings(), 100.0, true);
  EXPECT_NEAR(judgement.data_movement_share, LIMITING_SHARE_THRESHOLD, 1e-12);
  EXPECT_TRUE(judgement.limited_by == WorkLimit::data_movement);
}

void should_report_a_share_above_one_when_a_kernel_beats_the_calibration() {
  const WorkJudgement judgement = judge_work(work_of(1'000, 0, 150e6), round_ceilings(), 100.0, true);
  EXPECT_NEAR(judgement.arithmetic_share, 1.5, 1e-9);
  EXPECT_TRUE(judgement.limited_by == WorkLimit::arithmetic);
}

void should_price_work_but_not_judge_a_unit_without_a_gpu_window() {
  const WorkJudgement judgement = judge_work(work_of(1'000, 0, 0.0), round_ceilings(), 50.0, false);
  EXPECT_TRUE(judgement.has_data_movement_time);
  EXPECT_TRUE(!judgement.has_shares);
  EXPECT_TRUE(judgement.limited_by == WorkLimit::not_judged);
}

// --- negative ---

void should_not_judge_without_measured_ceilings() {
  const WorkJudgement judgement = judge_work(work_of(1'000, 0, 1e6), MachineCeilings{}, 100.0, true);
  EXPECT_TRUE(!judgement.has_data_movement_time && !judgement.has_shares);
  EXPECT_TRUE(judgement.limited_by == WorkLimit::not_judged);
}

void should_not_judge_a_unit_that_moved_nothing() {
  const WorkJudgement judgement = judge_work(NodeWork{}, round_ceilings(), 100.0, true);
  EXPECT_TRUE(judgement.limited_by == WorkLimit::not_judged);
  EXPECT_TRUE(work_limit_name(judgement.limited_by).empty());
}

void should_not_judge_a_zero_length_window() {
  const WorkJudgement judgement = judge_work(work_of(1'000, 0, 0.0), round_ceilings(), 0.0, true);
  EXPECT_TRUE(!judgement.has_shares);
}

void should_name_each_limit_for_the_measurement_file() {
  EXPECT_TRUE(work_limit_name(WorkLimit::data_movement) == "data_movement");
  EXPECT_TRUE(work_limit_name(WorkLimit::arithmetic) == "arithmetic");
  EXPECT_TRUE(work_limit_name(WorkLimit::neither) == "neither");
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_name_data_movement_when_the_bytes_fill_most_of_the_time),
      TEST_CASE(should_name_arithmetic_when_the_operations_fill_most_of_the_time),
      TEST_CASE(should_name_neither_when_both_needs_are_small),
      TEST_CASE(should_count_a_share_of_exactly_one_half_as_the_limit),
      TEST_CASE(should_report_a_share_above_one_when_a_kernel_beats_the_calibration),
      TEST_CASE(should_price_work_but_not_judge_a_unit_without_a_gpu_window),
      TEST_CASE(should_not_judge_without_measured_ceilings),
      TEST_CASE(should_not_judge_a_unit_that_moved_nothing),
      TEST_CASE(should_not_judge_a_zero_length_window),
      TEST_CASE(should_name_each_limit_for_the_measurement_file),
  });
}
