#pragma once

// Wire release probe: one concern only — measure, on this Mac, how long
// macOS keeps memory wired after MLX releases it (#45): wait for the wired
// total to settle, allocate and evaluate one array, release it as soon as the
// GPU is done and clear MLX's cache, then time from the first moment MLX's
// holding has fallen to the moment the system's wired total has fallen by at
// least half the allocation.
//
// Measured with live sysctl reads on the Mac #45 was built on, memory released right after
// its last GPU use (as here, and as an engine releases a finished prompt's
// memory) was un-wired about 0.2 s later. It varies by machine and load and
// is longer than many measured runs, so an engine sizing against "wired
// outside MLX" needs its own machine's figure. (Memory MLX still holds is
// un-wired too, after about 1.2 s without GPU use; that is not what the
// probe measures.)

#include <cstdint>
#include <optional>
#include <vector>

namespace mlx_perf {

struct WireReleaseProbeRequest {
  std::uint64_t allocation_bytes = 1ull << 30;  // rounded down to whole float32 elements
  int repetition_count = 5;
  // A repetition whose wired total has not fallen within this long counts
  // as unresolved.
  double unwire_wait_seconds = 5.0;
};

struct WireReleaseRepetition {
  // How far the system's wired total rose while MLX held the allocation.
  std::int64_t wired_rise_bytes = 0;
  // Empty when MLX's holding or the wired total had not fallen in time.
  std::optional<double> release_lag_microseconds;
};

struct WireReleaseProbeResult {
  std::uint64_t allocation_bytes = 0;
  std::uint64_t wire_limit_bytes = 0;  // 0 when unknown
  std::vector<WireReleaseRepetition> repetitions;
};

// Throws std::invalid_argument when allocation_bytes is under one float32
// element or 8 GiB and over (one MLX array), or repetition_count is below 1.
// Reports progress with an ETA.
WireReleaseProbeResult measure_wire_release_lag(const WireReleaseProbeRequest& probe_request);

// Over the repetitions whose lag resolved; empty when none did.
std::optional<double> median_probe_release_lag_microseconds(const WireReleaseProbeResult& probe_result);
std::optional<double> largest_probe_release_lag_microseconds(const WireReleaseProbeResult& probe_result);
int count_resolved_repetitions(const WireReleaseProbeResult& probe_result);
std::int64_t median_wired_rise_bytes(const WireReleaseProbeResult& probe_result);

} // namespace mlx_perf
