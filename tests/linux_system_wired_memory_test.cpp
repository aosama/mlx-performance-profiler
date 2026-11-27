// Unit tests for linux_system_wired_memory: Linux has no system-wide wired
// line, so the reading must say "unavailable" the way the header defines it
// (both fields 0), which the wired memory sections already understand.

#include "system_wired_memory.h"
#include "test_harness.h"

namespace {

// --- positive ---

void should_report_the_reading_unavailable() {
  const mlx_perf::SystemWiredMemoryReading wired_reading = mlx_perf::read_system_wired_memory();
  EXPECT_TRUE(wired_reading.wired_bytes == 0 && wired_reading.wire_limit_bytes == 0);
}

// --- edge / boundary ---

void should_report_no_wire_limit() {
  EXPECT_TRUE(mlx_perf::read_system_wired_memory().wire_limit_bytes == 0);
}

// --- negative ---

void should_not_claim_any_wired_memory() {
  EXPECT_TRUE(mlx_perf::read_system_wired_memory().wired_bytes == 0);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_report_the_reading_unavailable),
      TEST_CASE(should_report_no_wire_limit),
      TEST_CASE(should_not_claim_any_wired_memory),
  });
}
