#include "system_wired_memory.h"

namespace mlx_perf {

// Linux has no system-wide wired-memory line like macOS's
// vm.global_user_wire_limit, and a discrete GPU's memory is not host memory,
// so there is nothing honest to report: both fields stay 0, which the header
// defines as "the kernel refused the read", and the wired memory sections
// say the figures are unavailable rather than showing zeros as measured.
SystemWiredMemoryReading read_system_wired_memory() {
  return SystemWiredMemoryReading{};
}

} // namespace mlx_perf
