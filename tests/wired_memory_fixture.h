#pragma once

// Synthetic wired memory samples for the pure wired-memory tests: a run in
// which MLX holds 1 GiB, releases it, and macOS un-wires it some samples
// later. No GPU, no system calls.

#include "wired_memory_sample.h"

#include <cstdint>
#include <vector>

namespace wired_memory_fixture {

inline constexpr std::uint64_t GIB = 1ull << 30;
inline constexpr std::uint64_t LIMIT_BYTES = 40 * GIB;

// `outside_bytes` wired by others, `mlx_bytes` held by MLX, and
// `pending_unwire_bytes` that MLX released but macOS still wires.
inline mlx_perf::WiredMemorySample sample_at(double milliseconds, std::uint64_t outside_bytes,
                                             std::uint64_t mlx_bytes, std::uint64_t pending_unwire_bytes = 0,
                                             std::uint64_t limit_bytes = LIMIT_BYTES) {
  mlx_perf::WiredMemorySample wired_memory_sample;
  wired_memory_sample.host_seconds = 100.0 + milliseconds / 1e3;
  wired_memory_sample.sample_point = mlx_perf::WiredSamplePoint::interval;
  wired_memory_sample.system_wired_bytes = outside_bytes + mlx_bytes + pending_unwire_bytes;
  wired_memory_sample.wire_limit_bytes = limit_bytes;
  wired_memory_sample.mlx_active_bytes = mlx_bytes;
  return wired_memory_sample;
}

// 3 GiB wired by others throughout. MLX holds 1 GiB for samples 0–2,
// releases it at sample 3 (10 ms); macOS un-wires it at sample 6 (40 ms).
inline std::vector<mlx_perf::WiredMemorySample> release_then_unwire() {
  return {sample_at(0, 3 * GIB, GIB),     sample_at(5, 3 * GIB, GIB),        sample_at(8, 3 * GIB, GIB),
          sample_at(10, 3 * GIB, 0, GIB), sample_at(20, 3 * GIB, 0, GIB),    sample_at(30, 3 * GIB, 0, GIB / 2 + 1),
          sample_at(40, 3 * GIB, 0, 0),   sample_at(50, 3 * GIB, 0, 0)};
}

} // namespace wired_memory_fixture
