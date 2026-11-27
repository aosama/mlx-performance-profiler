#include "system_memory_pressure.h"

namespace mlx_perf {

// The three counters are macOS's (wire refusals over the global limit,
// compressor input, swap in use), and Linux has no wire limit to refuse
// against. Reported unreadable rather than as zeros, so a memory verdict
// never claims "no pressure" from figures it could not read.
MemoryPressureCounters read_memory_pressure_counters() {
  return MemoryPressureCounters{};
}

} // namespace mlx_perf
