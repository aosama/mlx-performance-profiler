#pragma once

// System memory pressure: one concern only — read the kernel's running
// counters of memory pressure, so a measured run can say whether it pushed
// the system over the wire limit or made it compress or swap memory (#46).
// No MLX.
//
// Why these counters: `vm.add_wire_count_over_global_limit` counts the wire
// requests macOS refused because the wired total would have gone over
// `vm.global_user_wire_limit`, the kernel's own record of crossing the line.
// `vm.compressor_input_bytes` counts the bytes the compressor has taken in.
// `vm.swapusage` gives the swap in use. All three are live sysctl reads: the
// compressor and swap figures in vm_statistics64 come through
// host_statistics64, which macOS rate limits to a few fresh answers a second.
// All three are system-wide, so another program's pressure shows too.

#include <cstdint>

namespace mlx_perf {

struct MemoryPressureCounters {
  std::uint64_t wire_requests_over_limit_count = 0;  // running total since boot
  std::uint64_t compressor_input_bytes = 0;          // running total since boot
  std::uint64_t swap_used_bytes = 0;                 // now
  bool is_readable = false;  // false when the kernel refused any of the reads
};

MemoryPressureCounters read_memory_pressure_counters();

} // namespace mlx_perf
