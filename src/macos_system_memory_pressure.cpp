#include "system_memory_pressure.h"

#include <sys/sysctl.h>

#include <cstddef>
#include <optional>

namespace mlx_perf {
namespace {

// Some counters are 32-bit and some 64-bit, depending on the counter and the
// macOS release: read into 64 bits and accept either width (little-endian,
// so a 32-bit value lands in the low half of the zeroed 64-bit value).
std::optional<std::uint64_t> read_counter(const char* counter_name) {
  std::uint64_t counter_value = 0;
  std::size_t value_size = sizeof(counter_value);
  if (sysctlbyname(counter_name, &counter_value, &value_size, nullptr, 0) != 0 ||
      (value_size != sizeof(std::uint32_t) && value_size != sizeof(std::uint64_t))) {
    return std::nullopt;
  }
  return counter_value;
}

std::optional<std::uint64_t> read_swap_used_bytes() {
  xsw_usage swap_usage{};
  std::size_t value_size = sizeof(swap_usage);
  if (sysctlbyname("vm.swapusage", &swap_usage, &value_size, nullptr, 0) != 0 || value_size != sizeof(swap_usage)) {
    return std::nullopt;
  }
  return swap_usage.xsu_used;
}

} // namespace

MemoryPressureCounters read_memory_pressure_counters() {
  const std::optional<std::uint64_t> over_limit_count = read_counter("vm.add_wire_count_over_global_limit");
  const std::optional<std::uint64_t> compressor_input_bytes = read_counter("vm.compressor_input_bytes");
  const std::optional<std::uint64_t> swap_used_bytes = read_swap_used_bytes();
  MemoryPressureCounters pressure_counters;
  pressure_counters.is_readable =
      over_limit_count.has_value() && compressor_input_bytes.has_value() && swap_used_bytes.has_value();
  pressure_counters.wire_requests_over_limit_count = over_limit_count.value_or(0);
  pressure_counters.compressor_input_bytes = compressor_input_bytes.value_or(0);
  pressure_counters.swap_used_bytes = swap_used_bytes.value_or(0);
  return pressure_counters;
}

} // namespace mlx_perf
