// Unit tests for work_record: the ceilings and a unit's work and judgement
// as JSON, with null (never 0) for what was not known. No GPU.

#include "json_writer.h"
#include "test_harness.h"
#include "work_record.h"

#include <string>

namespace {

using namespace mlx_perf;

MachineCeilings round_ceilings() {
  return {true, 1e9, 1e12, "bfloat16"};
}

NodeWork gathered_work() {
  NodeWork unit_work;
  unit_work.bytes_read = 3'000;
  unit_work.bytes_written = 1'000;
  unit_work.is_arithmetic_counted = true;
  unit_work.arithmetic_operations = 4e6;
  unit_work.is_gathered = true;
  unit_work.gathered_experts_total = 8;
  unit_work.gathered_experts_used = 3;
  return unit_work;
}

const JsonValue& field_of(const JsonValue& record, const std::string& field_name) {
  return record.as_object().at(field_name);
}

// --- positive ---

void should_write_a_judged_unit_with_every_figure() {
  const NodeWork unit_work = gathered_work();
  const JsonValue work_record = build_work_record(unit_work, judge_work(unit_work, round_ceilings(), 8.0, true));
  EXPECT_TRUE(field_of(work_record, "bytes_read").as_number() == 3'000);
  EXPECT_TRUE(field_of(work_record, "gathered_experts_used").as_number() == 3);
  EXPECT_NEAR(field_of(work_record, "data_movement_at_ceiling_us").as_number(), 4.0, 1e-9);
  EXPECT_NEAR(field_of(work_record, "arithmetic_share").as_number(), 0.5, 1e-9);
  EXPECT_TRUE(field_of(work_record, "limited_by").as_string() == "data_movement");
}

void should_write_the_measured_ceilings() {
  const JsonValue ceilings_record = build_machine_ceilings_record(round_ceilings());
  EXPECT_TRUE(field_of(ceilings_record, "memory_read_bytes_per_second").as_number() == 1e9);
  EXPECT_TRUE(field_of(ceilings_record, "arithmetic_dtype").as_string() == "bfloat16");
}

// --- edge / boundary ---

void should_null_the_shares_of_a_unit_without_a_gpu_window() {
  const NodeWork unit_work = gathered_work();
  const JsonValue work_record = build_work_record(unit_work, judge_work(unit_work, round_ceilings(), 0.0, false));
  EXPECT_TRUE(!field_of(work_record, "data_movement_at_ceiling_us").is_null());
  EXPECT_TRUE(field_of(work_record, "data_movement_share").is_null());
  EXPECT_TRUE(field_of(work_record, "limited_by").is_null());
}

void should_null_uncounted_arithmetic_and_unresolved_experts() {
  NodeWork unit_work = gathered_work();
  unit_work.is_arithmetic_counted = false;
  unit_work.gathered_experts_used = 0;
  const JsonValue work_record = build_work_record(unit_work, judge_work(unit_work, round_ceilings(), 8.0, true));
  EXPECT_TRUE(field_of(work_record, "arithmetic_operations").is_null());
  EXPECT_TRUE(field_of(work_record, "arithmetic_share").is_null());
  EXPECT_TRUE(field_of(work_record, "gathered_experts_used").is_null());
  EXPECT_TRUE(field_of(work_record, "gathered_experts_total").as_number() == 8);
}

// --- negative ---

void should_write_null_for_a_unit_that_moved_nothing() {
  EXPECT_TRUE(build_work_record(NodeWork{}, WorkJudgement{}).is_null());
}

void should_write_null_ceilings_when_not_measured() {
  EXPECT_TRUE(build_machine_ceilings_record(MachineCeilings{}).is_null());
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_write_a_judged_unit_with_every_figure),
      TEST_CASE(should_write_the_measured_ceilings),
      TEST_CASE(should_null_the_shares_of_a_unit_without_a_gpu_window),
      TEST_CASE(should_null_uncounted_arithmetic_and_unresolved_experts),
      TEST_CASE(should_write_null_for_a_unit_that_moved_nothing),
      TEST_CASE(should_write_null_ceilings_when_not_measured),
  });
}
