#include "gpu_host_clock.h"

#include <mach/mach_time.h>

namespace mlx_perf {
namespace {

mach_timebase_info_data_t query_timebase() {
  mach_timebase_info_data_t queried_timebase{};
  mach_timebase_info(&queried_timebase);
  return queried_timebase;
}

} // namespace

double current_gpu_host_seconds() {
  static const mach_timebase_info_data_t timebase = query_timebase();
  const double nanoseconds = static_cast<double>(mach_absolute_time()) * timebase.numer / timebase.denom;
  return nanoseconds / 1e9;
}

} // namespace mlx_perf
