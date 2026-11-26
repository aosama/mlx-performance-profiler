// Integration tests for system_memory_pressure against this Mac's kernel:
// the counters must be readable and behave like running counters. No GPU.

#include "system_memory_pressure.h"
#include "test_harness.h"

namespace {

using mlx_perf::MemoryPressureCounters;
using mlx_perf::read_memory_pressure_counters;

// --- positive ---

void should_read_every_counter() {
  const MemoryPressureCounters pressure_counters = read_memory_pressure_counters();
  EXPECT_TRUE(pressure_counters.is_readable);
}

// --- edge / boundary ---

void should_never_see_a_running_counter_go_backwards() {
  const MemoryPressureCounters first_reading = read_memory_pressure_counters();
  const MemoryPressureCounters second_reading = read_memory_pressure_counters();
  EXPECT_TRUE(second_reading.compressor_input_bytes >= first_reading.compressor_input_bytes);
  EXPECT_TRUE(second_reading.wire_requests_over_limit_count >= first_reading.wire_requests_over_limit_count);
}

// --- negative ---

// Swap in use is a level, not a running total: it cannot exceed what a Mac
// could plausibly configure (1 TiB), which a misread struct would.
void should_not_misread_swap_in_use() {
  EXPECT_TRUE(read_memory_pressure_counters().swap_used_bytes < (1ull << 40));
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_read_every_counter),
      TEST_CASE(should_never_see_a_running_counter_go_backwards),
      TEST_CASE(should_not_misread_swap_in_use),
  });
}
