#include "system_wired_memory.h"

#include <mach/mach.h>
#include <sys/sysctl.h>

#include <cstddef>

namespace mlx_perf {
namespace {

// Preferred: the kernel's own counter through sysctl, which is read live on
// every call. host_statistics64 is rate limited for ordinary processes (a
// handful of fresh answers a second, cached figures after that), which made
// fine sampling read stale totals; it stays only as a fallback.
std::uint64_t read_wired_bytes_from_sysctl() {
  unsigned int wired_page_count = 0;
  std::size_t value_size = sizeof(wired_page_count);
  if (sysctlbyname("vm.page_wired_count", &wired_page_count, &value_size, nullptr, 0) != 0 ||
      value_size != sizeof(wired_page_count)) {
    return 0;
  }
  return static_cast<std::uint64_t>(wired_page_count) * vm_page_size;
}

std::uint64_t read_wired_bytes_from_host_statistics() {
  vm_statistics64_data_t virtual_memory_statistics{};
  mach_msg_type_number_t statistics_count = HOST_VM_INFO64_COUNT;
  const kern_return_t read_status =
      host_statistics64(mach_host_self(), HOST_VM_INFO64,
                        reinterpret_cast<host_info64_t>(&virtual_memory_statistics), &statistics_count);
  if (read_status != KERN_SUCCESS) {
    return 0;
  }
  return static_cast<std::uint64_t>(virtual_memory_statistics.wire_count) * vm_page_size;
}

std::uint64_t read_wire_limit_bytes() {
  std::uint64_t wire_limit_bytes = 0;
  std::size_t value_size = sizeof(wire_limit_bytes);
  if (sysctlbyname("vm.global_user_wire_limit", &wire_limit_bytes, &value_size, nullptr, 0) != 0 ||
      value_size != sizeof(wire_limit_bytes)) {
    return 0;
  }
  return wire_limit_bytes;
}

} // namespace

SystemWiredMemoryReading read_system_wired_memory() {
  const std::uint64_t sysctl_wired_bytes = read_wired_bytes_from_sysctl();
  const std::uint64_t wired_bytes =
      sysctl_wired_bytes > 0 ? sysctl_wired_bytes : read_wired_bytes_from_host_statistics();
  return SystemWiredMemoryReading{wired_bytes, read_wire_limit_bytes()};
}

} // namespace mlx_perf
