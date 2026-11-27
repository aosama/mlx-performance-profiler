// Unit tests for node_work: a stage's work is the sum of its members'.
// Synthetic numbers, no GPU.

#include "node_work.h"
#include "test_harness.h"

namespace {

using namespace mlx_perf;

NodeWork work_of(std::uint64_t bytes_read, std::uint64_t bytes_written, double operations) {
  NodeWork unit_work;
  unit_work.bytes_read = bytes_read;
  unit_work.bytes_written = bytes_written;
  unit_work.is_arithmetic_counted = operations > 0.0;
  unit_work.arithmetic_operations = operations;
  return unit_work;
}

// --- positive ---

void should_add_member_bytes_and_arithmetic_into_a_stage_total() {
  NodeWork stage_work;
  add_node_work(stage_work, work_of(100, 10, 5e6));
  add_node_work(stage_work, work_of(200, 20, 7e6));
  EXPECT_TRUE(stage_work.bytes_read == 300 && stage_work.bytes_written == 30);
  EXPECT_TRUE(stage_work.is_arithmetic_counted);
  EXPECT_NEAR(stage_work.arithmetic_operations, 12e6, 1e-9);
}

// --- edge / boundary ---

void should_count_a_stage_arithmetic_when_only_some_members_counted_theirs() {
  NodeWork stage_work;
  add_node_work(stage_work, work_of(100, 10, 0.0));
  add_node_work(stage_work, work_of(200, 20, 5e6));
  EXPECT_TRUE(stage_work.is_arithmetic_counted);
  EXPECT_NEAR(stage_work.arithmetic_operations, 5e6, 1e-9);
}

void should_keep_a_stage_uncounted_when_no_member_counted_arithmetic() {
  NodeWork stage_work;
  add_node_work(stage_work, work_of(100, 10, 0.0));
  EXPECT_TRUE(!stage_work.is_arithmetic_counted);
  EXPECT_TRUE(stage_work.arithmetic_operations == 0.0);
}

// --- negative ---

void should_leave_a_total_unchanged_when_adding_no_work() {
  NodeWork stage_work = work_of(100, 10, 5e6);
  add_node_work(stage_work, NodeWork{});
  EXPECT_TRUE(stage_work.bytes_read == 100 && stage_work.bytes_written == 10);
  EXPECT_NEAR(stage_work.arithmetic_operations, 5e6, 1e-9);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_add_member_bytes_and_arithmetic_into_a_stage_total),
      TEST_CASE(should_count_a_stage_arithmetic_when_only_some_members_counted_theirs),
      TEST_CASE(should_keep_a_stage_uncounted_when_no_member_counted_arithmetic),
      TEST_CASE(should_leave_a_total_unchanged_when_adding_no_work),
  });
}
