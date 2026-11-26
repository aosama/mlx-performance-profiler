#include "wired_memory_sample.h"

namespace mlx_perf {

std::string_view wired_sample_point_name(WiredSamplePoint sample_point) {
  switch (sample_point) {
    case WiredSamplePoint::run_start: return "run_start";
    case WiredSamplePoint::evaluation_boundary: return "evaluation_boundary";
    case WiredSamplePoint::interval: return "interval";
    case WiredSamplePoint::run_end: return "run_end";
    case WiredSamplePoint::after_run: return "after_run";
  }
  return "interval";
}

std::uint64_t mlx_held_bytes(const WiredMemorySample& wired_memory_sample) {
  return wired_memory_sample.mlx_active_bytes + wired_memory_sample.mlx_cache_bytes;
}

std::int64_t wired_outside_mlx_bytes(const WiredMemorySample& wired_memory_sample) {
  return static_cast<std::int64_t>(wired_memory_sample.system_wired_bytes) -
         static_cast<std::int64_t>(mlx_held_bytes(wired_memory_sample));
}

std::int64_t wire_headroom_bytes(const WiredMemorySample& wired_memory_sample) {
  if (wired_memory_sample.wire_limit_bytes == 0) {
    return 0;
  }
  return static_cast<std::int64_t>(wired_memory_sample.wire_limit_bytes) -
         static_cast<std::int64_t>(wired_memory_sample.system_wired_bytes);
}

} // namespace mlx_perf
