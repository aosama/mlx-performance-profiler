// Integration tests for system_wired_memory against this Mac's kernel: the
// readings must be plausible and consistent, never exact. No GPU.

#include "system_wired_memory.h"
#include "test_harness.h"

#include <sys/sysctl.h>

#include <cstdint>

namespace {

using mlx_perf::read_system_wired_memory;
using mlx_perf::SystemWiredMemoryReading;

std::uint64_t physical_memory_bytes() {
  std::uint64_t memory_bytes = 0;
  std::size_t value_size = sizeof(memory_bytes);
  sysctlbyname("hw.memsize", &memory_bytes, &value_size, nullptr, 0);
  return memory_bytes;
}

// --- positive ---

void should_read_wired_memory_and_a_limit_below_physical_memory() {
  const SystemWiredMemoryReading reading = read_system_wired_memory();
  EXPECT_TRUE(reading.wired_bytes > 0);
  EXPECT_TRUE(reading.wire_limit_bytes > 0);
  EXPECT_TRUE(reading.wire_limit_bytes < physical_memory_bytes());
  EXPECT_TRUE(reading.wired_bytes < physical_memory_bytes());
}

// --- edge / boundary ---

void should_read_whole_pages() {
  EXPECT_TRUE(read_system_wired_memory().wired_bytes % 4096 == 0);
}

void should_read_the_same_limit_twice_in_a_row() {
  EXPECT_TRUE(read_system_wired_memory().wire_limit_bytes == read_system_wired_memory().wire_limit_bytes);
}

// --- negative ---

// The limit is a ceiling the system stays under in normal operation: a
// reading above it would mean the two figures were mixed up.
void should_not_read_more_wired_than_the_limit_on_an_idle_machine() {
  const SystemWiredMemoryReading reading = read_system_wired_memory();
  EXPECT_TRUE(reading.wired_bytes < reading.wire_limit_bytes);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_read_wired_memory_and_a_limit_below_physical_memory),
      TEST_CASE(should_read_whole_pages),
      TEST_CASE(should_read_the_same_limit_twice_in_a_row),
      TEST_CASE(should_not_read_more_wired_than_the_limit_on_an_idle_machine),
  });
}
