#pragma once

// System wired memory: one concern only — read, from the kernel, how much
// memory is wired system-wide and the line macOS keeps it under. No MLX.
//
// Why these two: on Apple silicon the memory the GPU works on is wired, and
// macOS keeps wired memory under a system-wide limit
// (`vm.global_user_wire_limit`). How much room is left under that line is
// what decides whether the GPU's memory stays wired (#45). Both reads are
// cheap system calls (a few microseconds), cheap enough for every
// evaluation boundary. The wired total comes from sysctl, not
// host_statistics64, which macOS rate limits to a few fresh answers a second.

#include <cstdint>

namespace mlx_perf {

struct SystemWiredMemoryReading {
  std::uint64_t wired_bytes = 0;       // vm.page_wired_count × page size
  std::uint64_t wire_limit_bytes = 0;  // vm.global_user_wire_limit; 0 when unreadable
};

// Read fresh every call: an administrator can change the limit at run time.
// Both fields are 0 when the kernel refuses the read.
SystemWiredMemoryReading read_system_wired_memory();

} // namespace mlx_perf
