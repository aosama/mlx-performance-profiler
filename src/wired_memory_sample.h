#pragma once

// Wired memory sample: one concern only — one instant's reading of the
// system's wired memory beside MLX's own, and the figures derived from it.
// Pure: taking a sample (system calls, MLX) lives in wired_memory_sampler.
//
// Why "outside MLX" is derived, not read: it is what an engine computes to
// size what it keeps in memory (wired total less MLX's active memory and
// buffer cache). Recording its inputs at one instant lets a reader see when
// that figure is misled, for example by macOS un-wiring memory a while after
// MLX stopped counting it (#45).

#include <cstdint>
#include <string_view>

namespace mlx_perf {

// Where in a measurement the sample was taken.
enum class WiredSamplePoint {
  run_start,            // before the measured code started
  evaluation_boundary,  // right after a blocking wait for the GPU returned
  interval,             // the fine interval sampler, between boundaries
  run_end,              // after the measured code's GPU work completed
  after_run,            // following an MLX release until macOS un-wired it
};

std::string_view wired_sample_point_name(WiredSamplePoint sample_point);

struct WiredMemorySample {
  double host_seconds = 0.0;  // the GPU host clock (gpu_host_clock)
  WiredSamplePoint sample_point = WiredSamplePoint::interval;
  std::uint64_t system_wired_bytes = 0;
  std::uint64_t wire_limit_bytes = 0;  // 0 when unknown
  std::uint64_t mlx_active_bytes = 0;
  std::uint64_t mlx_cache_bytes = 0;
};

// MLX's whole holding: active arrays plus its buffer cache.
std::uint64_t mlx_held_bytes(const WiredMemorySample& wired_memory_sample);

// The system's wired total less MLX's holding. Signed: memory MLX counts
// that macOS has not wired yet makes it negative.
std::int64_t wired_outside_mlx_bytes(const WiredMemorySample& wired_memory_sample);

// The wire limit less the system's wired total: negative once over the line.
// Meaningless (and 0) when the limit is unknown.
std::int64_t wire_headroom_bytes(const WiredMemorySample& wired_memory_sample);

} // namespace mlx_perf
