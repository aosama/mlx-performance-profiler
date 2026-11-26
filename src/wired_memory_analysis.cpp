#include "wired_memory_analysis.h"

#include "sample_statistics.h"

#include <algorithm>
#include <set>

namespace mlx_perf {
namespace {

using Samples = std::vector<WiredMemorySample>;

void find_peaks(const Samples& wired_memory_samples, WiredMemoryAnalysis& wired_memory_analysis) {
  wired_memory_analysis.peak_wired_outside_mlx_bytes = wired_outside_mlx_bytes(wired_memory_samples.front());
  wired_memory_analysis.lowest_wired_outside_mlx_bytes = wired_memory_analysis.peak_wired_outside_mlx_bytes;
  for (const WiredMemorySample& wired_memory_sample : wired_memory_samples) {
    wired_memory_analysis.peak_system_wired_bytes =
        std::max(wired_memory_analysis.peak_system_wired_bytes, wired_memory_sample.system_wired_bytes);
    wired_memory_analysis.peak_mlx_held_bytes =
        std::max(wired_memory_analysis.peak_mlx_held_bytes, mlx_held_bytes(wired_memory_sample));
    wired_memory_analysis.peak_wired_outside_mlx_bytes =
        std::max(wired_memory_analysis.peak_wired_outside_mlx_bytes, wired_outside_mlx_bytes(wired_memory_sample));
    wired_memory_analysis.lowest_wired_outside_mlx_bytes =
        std::min(wired_memory_analysis.lowest_wired_outside_mlx_bytes, wired_outside_mlx_bytes(wired_memory_sample));
    if (wired_memory_sample.wire_limit_bytes == 0) {
      continue;
    }
    wired_memory_analysis.wire_limit_bytes = wired_memory_sample.wire_limit_bytes;
    const std::int64_t headroom_bytes = wire_headroom_bytes(wired_memory_sample);
    wired_memory_analysis.lowest_wire_headroom_bytes =
        std::min(wired_memory_analysis.lowest_wire_headroom_bytes.value_or(headroom_bytes), headroom_bytes);
    wired_memory_analysis.has_crossed_wire_limit |= headroom_bytes < 0;
  }
}

// Wired bytes at or below which macOS counts as having followed a release:
// half the released amount below where the system stood just before it.
std::int64_t unwired_threshold_bytes(const WiredMemorySample& before_release, std::uint64_t released_bytes) {
  return static_cast<std::int64_t>(before_release.system_wired_bytes) -
         static_cast<std::int64_t>(released_bytes / 2);
}

WireReleaseEvent follow_release(const Samples& wired_memory_samples, int released_sample_index) {
  const WiredMemorySample& before_release = wired_memory_samples[released_sample_index - 1];
  const WiredMemorySample& at_release = wired_memory_samples[released_sample_index];
  WireReleaseEvent release_event;
  release_event.released_sample_index = released_sample_index;
  release_event.released_bytes = mlx_held_bytes(before_release) - mlx_held_bytes(at_release);
  const std::int64_t threshold_bytes = unwired_threshold_bytes(before_release, release_event.released_bytes);
  for (int sample_index = released_sample_index; sample_index < static_cast<int>(wired_memory_samples.size());
       ++sample_index) {
    const WiredMemorySample& later_sample = wired_memory_samples[sample_index];
    if (static_cast<std::int64_t>(later_sample.system_wired_bytes) <= threshold_bytes) {
      release_event.unwired_sample_index = sample_index;
      release_event.release_lag_microseconds = (later_sample.host_seconds - at_release.host_seconds) * 1e6;
      break;
    }
  }
  return release_event;
}

void find_release_events(const Samples& wired_memory_samples, WiredMemoryAnalysis& wired_memory_analysis) {
  for (int sample_index = 1; sample_index < static_cast<int>(wired_memory_samples.size()); ++sample_index) {
    const std::uint64_t held_before = mlx_held_bytes(wired_memory_samples[sample_index - 1]);
    const std::uint64_t held_now = mlx_held_bytes(wired_memory_samples[sample_index]);
    if (held_before > held_now && held_before - held_now >= RELEASE_EVENT_MINIMUM_BYTES) {
      wired_memory_analysis.release_events.push_back(follow_release(wired_memory_samples, sample_index));
    }
  }
}

// From a release until macOS follows (or the samples end), the system still
// wires memory MLX no longer counts: the part above where it will settle.
void mark_inflated_samples(const Samples& wired_memory_samples, WiredMemoryAnalysis& wired_memory_analysis) {
  std::set<int> inflated_sample_indices;
  for (const WireReleaseEvent& release_event : wired_memory_analysis.release_events) {
    const WiredMemorySample& before_release = wired_memory_samples[release_event.released_sample_index - 1];
    const std::int64_t settled_wired_bytes = static_cast<std::int64_t>(before_release.system_wired_bytes) -
                                             static_cast<std::int64_t>(release_event.released_bytes);
    const int end_index = release_event.unwired_sample_index.value_or(static_cast<int>(wired_memory_samples.size()));
    for (int sample_index = release_event.released_sample_index; sample_index < end_index; ++sample_index) {
      const std::int64_t still_wired_bytes =
          static_cast<std::int64_t>(wired_memory_samples[sample_index].system_wired_bytes) - settled_wired_bytes;
      const std::uint64_t inflation_bytes = static_cast<std::uint64_t>(std::clamp<std::int64_t>(
          still_wired_bytes, 0, static_cast<std::int64_t>(release_event.released_bytes)));
      inflated_sample_indices.insert(sample_index);
      wired_memory_analysis.largest_release_lag_inflation_bytes =
          std::max(wired_memory_analysis.largest_release_lag_inflation_bytes, inflation_bytes);
    }
  }
  wired_memory_analysis.release_lag_inflated_sample_indices.assign(inflated_sample_indices.begin(),
                                                                   inflated_sample_indices.end());
}

bool returns_within_two_samples(const Samples& wired_memory_samples, int jump_index, std::int64_t level_before) {
  const int last_index = std::min(jump_index + 2, static_cast<int>(wired_memory_samples.size()) - 1);
  const std::int64_t tolerance_bytes = static_cast<std::int64_t>(RELEASE_EVENT_MINIMUM_BYTES / 2);
  for (int sample_index = jump_index + 1; sample_index <= last_index; ++sample_index) {
    if (wired_outside_mlx_bytes(wired_memory_samples[sample_index]) <= level_before + tolerance_bytes) {
      return true;
    }
  }
  return false;
}

void find_transient_jumps(const Samples& wired_memory_samples, WiredMemoryAnalysis& wired_memory_analysis) {
  const std::int64_t minimum_jump_bytes = static_cast<std::int64_t>(RELEASE_EVENT_MINIMUM_BYTES);
  for (int sample_index = 1; sample_index < static_cast<int>(wired_memory_samples.size()); ++sample_index) {
    const std::int64_t level_before = wired_outside_mlx_bytes(wired_memory_samples[sample_index - 1]);
    const std::int64_t level_now = wired_outside_mlx_bytes(wired_memory_samples[sample_index]);
    if (level_now - level_before >= minimum_jump_bytes &&
        returns_within_two_samples(wired_memory_samples, sample_index, level_before)) {
      wired_memory_analysis.transient_jump_sample_indices.push_back(sample_index);
    }
  }
}

std::vector<double> resolved_lags(const std::vector<WireReleaseEvent>& release_events) {
  std::vector<double> lag_microseconds;
  for (const WireReleaseEvent& release_event : release_events) {
    if (release_event.release_lag_microseconds.has_value()) {
      lag_microseconds.push_back(*release_event.release_lag_microseconds);
    }
  }
  return lag_microseconds;
}

} // namespace

WiredMemoryAnalysis analyze_wired_memory(const std::vector<WiredMemorySample>& wired_memory_samples) {
  WiredMemoryAnalysis wired_memory_analysis;
  wired_memory_analysis.sample_count = static_cast<int>(wired_memory_samples.size());
  if (wired_memory_samples.empty()) {
    return wired_memory_analysis;
  }
  find_peaks(wired_memory_samples, wired_memory_analysis);
  find_release_events(wired_memory_samples, wired_memory_analysis);
  mark_inflated_samples(wired_memory_samples, wired_memory_analysis);
  find_transient_jumps(wired_memory_samples, wired_memory_analysis);
  return wired_memory_analysis;
}

std::optional<double> median_release_lag_microseconds(const std::vector<WireReleaseEvent>& release_events) {
  const std::vector<double> lag_microseconds = resolved_lags(release_events);
  if (lag_microseconds.empty()) {
    return std::nullopt;
  }
  return summarize_samples(lag_microseconds).median;
}

std::optional<double> largest_release_lag_microseconds(const std::vector<WireReleaseEvent>& release_events) {
  const std::vector<double> lag_microseconds = resolved_lags(release_events);
  if (lag_microseconds.empty()) {
    return std::nullopt;
  }
  return *std::max_element(lag_microseconds.begin(), lag_microseconds.end());
}

} // namespace mlx_perf
