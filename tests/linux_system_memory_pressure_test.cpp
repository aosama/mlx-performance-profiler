// Unit tests for linux_system_memory_pressure: with no wire limit to refuse
// against, the macOS counters must read as unreadable, never as zeros a
// memory verdict would take for "no pressure".

#include "system_memory_pressure.h"
#include "test_harness.h"

namespace {

// --- positive ---

void should_report_the_pressure_counters_unreadable() {
  EXPECT_TRUE(!mlx_perf::read_memory_pressure_counters().is_readable);
}

// --- edge / boundary ---

void should_give_the_same_answer_on_every_read() {
  EXPECT_TRUE(mlx_perf::read_memory_pressure_counters().is_readable ==
              mlx_perf::read_memory_pressure_counters().is_readable);
}

// --- negative ---

void should_claim_no_wire_refusals_compression_or_swap() {
  const mlx_perf::MemoryPressureCounters pressure_counters = mlx_perf::read_memory_pressure_counters();
  EXPECT_TRUE(pressure_counters.wire_requests_over_limit_count == 0);
  EXPECT_TRUE(pressure_counters.compressor_input_bytes == 0);
  EXPECT_TRUE(pressure_counters.swap_used_bytes == 0);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_report_the_pressure_counters_unreadable),
      TEST_CASE(should_give_the_same_answer_on_every_read),
      TEST_CASE(should_claim_no_wire_refusals_compression_or_swap),
  });
}
