#include "wired_memory_sampler.h"

#include "metal_host_clock.h"
#include "system_wired_memory.h"
#include "wired_memory_analysis.h"

#include "mlx/memory.h"

#include <algorithm>

namespace mlx_perf {
namespace {

void sort_by_time(std::vector<WiredMemorySample>& wired_memory_samples) {
  std::stable_sort(wired_memory_samples.begin(), wired_memory_samples.end(),
                   [](const WiredMemorySample& earlier, const WiredMemorySample& later) {
                     return earlier.host_seconds < later.host_seconds;
                   });
}

bool has_unresolved_release(const std::vector<WiredMemorySample>& wired_memory_samples) {
  for (const WireReleaseEvent& release_event : analyze_wired_memory(wired_memory_samples).release_events) {
    if (!release_event.unwired_sample_index.has_value()) {
      return true;
    }
  }
  return false;
}

} // namespace

WiredMemorySample take_wired_memory_sample(WiredSamplePoint sample_point) {
  WiredMemorySample wired_memory_sample;
  wired_memory_sample.sample_point = sample_point;
  // MLX's counters first and the system's last, then the clock: a release
  // between the reads shows as the system lagging MLX, which it really does.
  wired_memory_sample.mlx_active_bytes = mlx::core::get_active_memory();
  wired_memory_sample.mlx_cache_bytes = mlx::core::get_cache_memory();
  const SystemWiredMemoryReading system_reading = read_system_wired_memory();
  wired_memory_sample.system_wired_bytes = system_reading.wired_bytes;
  wired_memory_sample.wire_limit_bytes = system_reading.wire_limit_bytes;
  wired_memory_sample.host_seconds = current_metal_host_seconds();
  return wired_memory_sample;
}

WiredMemorySampler::WiredMemorySampler(std::chrono::microseconds sample_interval,
                                       std::chrono::milliseconds unwire_wait_limit)
    : sample_interval_(sample_interval), unwire_wait_limit_(unwire_wait_limit) {}

WiredMemorySampler::~WiredMemorySampler() {
  should_keep_sampling_ = false;
  if (interval_thread_.joinable()) {
    interval_thread_.join();
  }
}

void WiredMemorySampler::start() {
  add_sample(take_wired_memory_sample(WiredSamplePoint::run_start));
  should_keep_sampling_ = true;
  interval_thread_ = std::thread(&WiredMemorySampler::sample_on_interval, this);
}

void WiredMemorySampler::on_evaluation_boundary() {
  add_sample(take_wired_memory_sample(WiredSamplePoint::evaluation_boundary));
}

std::vector<WiredMemorySample> WiredMemorySampler::finish() {
  should_keep_sampling_ = false;
  if (interval_thread_.joinable()) {
    interval_thread_.join();
  }
  add_sample(take_wired_memory_sample(WiredSamplePoint::run_end));
  sample_until_unwired();
  const std::lock_guard<std::mutex> samples_lock(samples_mutex_);
  sort_by_time(wired_memory_samples_);
  return wired_memory_samples_;
}

void WiredMemorySampler::sample_on_interval() {
  while (should_keep_sampling_) {
    std::this_thread::sleep_for(sample_interval_);
    if (should_keep_sampling_) {
      add_sample(take_wired_memory_sample(WiredSamplePoint::interval));
    }
  }
}

void WiredMemorySampler::add_sample(const WiredMemorySample& wired_memory_sample) {
  const std::lock_guard<std::mutex> samples_lock(samples_mutex_);
  wired_memory_samples_.push_back(wired_memory_sample);
}

// Runs on the caller's thread after the run. A late boundary from a worker
// thread may still add a sample meanwhile, so the list is only touched under
// its lock; the release check works on a sorted copy.
void WiredMemorySampler::sample_until_unwired() {
  const auto wait_ends_at = std::chrono::steady_clock::now() + unwire_wait_limit_;
  while (std::chrono::steady_clock::now() < wait_ends_at) {
    std::vector<WiredMemorySample> samples_so_far;
    {
      const std::lock_guard<std::mutex> samples_lock(samples_mutex_);
      samples_so_far = wired_memory_samples_;
    }
    sort_by_time(samples_so_far);
    if (!has_unresolved_release(samples_so_far)) {
      return;
    }
    std::this_thread::sleep_for(sample_interval_);
    add_sample(take_wired_memory_sample(WiredSamplePoint::after_run));
  }
}

} // namespace mlx_perf
