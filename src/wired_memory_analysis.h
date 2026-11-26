#pragma once

// Wired memory analysis: one concern only — read a run's wired memory
// samples for what an engine sizing its memory needs to know: the peaks, the
// room left under the wire limit, when MLX released memory and how long macOS
// took to un-wire it, and which samples' "wired outside MLX" figure that
// delay inflated. Pure arithmetic over samples.
//
// What drives the wired total (measured for #45 on one Apple silicon Mac, with live
// sysctl reads): macOS wires memory when the GPU uses it. Memory MLX releases
// right after the GPU used it is un-wired about 0.2 s later; memory MLX
// still holds is un-wired after about 1.2 s without GPU use, and wired again
// the moment the GPU uses it. So "wired outside MLX" errs both ways: too high
// for a fraction of a second after a release (the release lag found here),
// and too low once memory MLX holds has sat idle (the lowest figure below),
// though that memory counts against the same limit again on its next use.
//
// Why release lag is measured from a sample, not from a free: MLX stops
// counting a buffer when the buffer is really released, which can be after
// the program dropped it (the GPU may still be using it). The first sample
// that sees MLX's holding fall is the earliest moment an engine could see the
// drop. So a lag is resolved only to the sampling interval, and a lag longer
// than the sampling window is reported as unresolved, never guessed.

#include "wired_memory_sample.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace mlx_perf {

// Smaller drops in MLX's holding are allocator churn, not releases an engine
// sizes against; 64 MiB is far below the gigabyte releases that mislead it.
inline constexpr std::uint64_t RELEASE_EVENT_MINIMUM_BYTES = 64ull << 20;

// A fall in MLX's holding between two samples, and when macOS followed.
struct WireReleaseEvent {
  int released_sample_index = 0;     // the first sample that saw MLX's holding fall
  std::uint64_t released_bytes = 0;  // how far it fell
  // The first sample at or after it whose system wired total had fallen by
  // at least half the released bytes; empty when no sample did.
  std::optional<int> unwired_sample_index;
  std::optional<double> release_lag_microseconds;
};

struct WiredMemoryAnalysis {
  int sample_count = 0;
  std::uint64_t wire_limit_bytes = 0;  // the last known limit; 0 when unknown
  std::uint64_t peak_system_wired_bytes = 0;
  std::uint64_t peak_mlx_held_bytes = 0;
  std::int64_t peak_wired_outside_mlx_bytes = 0;
  // Falls below what others really wire when memory MLX holds has gone idle
  // on the GPU and been un-wired (negative when that exceeds others' share).
  std::int64_t lowest_wired_outside_mlx_bytes = 0;
  std::optional<std::int64_t> lowest_wire_headroom_bytes;  // empty when the limit is unknown
  bool has_crossed_wire_limit = false;
  std::vector<WireReleaseEvent> release_events;
  // Samples taken while a release was still wired: their "outside MLX"
  // counts memory MLX had already let go of, by up to the amount shown.
  std::vector<int> release_lag_inflated_sample_indices;
  std::uint64_t largest_release_lag_inflation_bytes = 0;
  // Samples where "outside MLX" jumped by at least RELEASE_EVENT_MINIMUM_BYTES
  // and came back within the next two samples: the shape an engine sees.
  std::vector<int> transient_jump_sample_indices;
};

// `wired_memory_samples` in time order. Empty input gives an empty analysis.
WiredMemoryAnalysis analyze_wired_memory(const std::vector<WiredMemorySample>& wired_memory_samples);

// Median and largest resolved lag over the events; empty when none resolved.
std::optional<double> median_release_lag_microseconds(const std::vector<WireReleaseEvent>& release_events);
std::optional<double> largest_release_lag_microseconds(const std::vector<WireReleaseEvent>& release_events);

} // namespace mlx_perf
