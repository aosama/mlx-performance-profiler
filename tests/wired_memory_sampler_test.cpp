// Integration tests for wired_memory_sampler on the Metal GPU: samples at the
// run's start and end, at evaluation boundaries, on the interval, and after
// the run until macOS has un-wired what MLX released.

#include "submission_recorder.h"
#include "backend_expectations.h"
#include "test_harness.h"
#include "wired_memory_analysis.h"
#include "wired_memory_sampler.h"

#include "mlx/mlx.h"

#include <chrono>
#include <optional>
#include <thread>

namespace {

using namespace mlx_perf;

int count_samples_at(const std::vector<WiredMemorySample>& wired_memory_samples, WiredSamplePoint sample_point) {
  int sample_count = 0;
  for (const WiredMemorySample& wired_memory_sample : wired_memory_samples) {
    sample_count += wired_memory_sample.sample_point == sample_point ? 1 : 0;
  }
  return sample_count;
}

bool is_in_time_order(const std::vector<WiredMemorySample>& wired_memory_samples) {
  for (std::size_t sample_index = 1; sample_index < wired_memory_samples.size(); ++sample_index) {
    if (wired_memory_samples[sample_index].host_seconds < wired_memory_samples[sample_index - 1].host_seconds) {
      return false;
    }
  }
  return true;
}

// --- positive ---

void should_sample_each_evaluation_boundary_and_the_interval_between() {
  install_submission_recorder();
  WiredMemorySampler wired_memory_sampler(std::chrono::microseconds(500), std::chrono::milliseconds(0));
  SubmissionRecordingScope submission_scope;
  submission_scope.set_evaluation_boundary_listener(&wired_memory_sampler);
  wired_memory_sampler.start();
  mlx::core::array running_value = mlx::core::ones({256, 256});
  for (int step_index = 0; step_index < 4; ++step_index) {
    running_value = running_value + 1.0f;
    mlx::core::eval(running_value);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  submission_scope.set_evaluation_boundary_listener(nullptr);
  const std::vector<WiredMemorySample> wired_memory_samples = wired_memory_sampler.finish();
  EXPECT_TRUE(count_samples_at(wired_memory_samples, WiredSamplePoint::run_start) == 1);
  EXPECT_TRUE(count_samples_at(wired_memory_samples, WiredSamplePoint::evaluation_boundary) >= 4);
  EXPECT_TRUE(count_samples_at(wired_memory_samples, WiredSamplePoint::interval) >= 3);
  EXPECT_TRUE(count_samples_at(wired_memory_samples, WiredSamplePoint::run_end) == 1);
  EXPECT_TRUE(is_in_time_order(wired_memory_samples));
  EXPECT_TRUE(backend_expectations::HAS_SYSTEM_WIRED_MEMORY ? wired_memory_samples.front().system_wired_bytes > 0
                                                         : wired_memory_samples.front().system_wired_bytes == 0);
}

// The behaviour #45 is about: a release, then macOS un-wiring it later.
void should_keep_sampling_after_the_run_until_macos_unwires_a_release() {
  WiredMemorySampler wired_memory_sampler;
  wired_memory_sampler.start();
  std::optional<mlx::core::array> large_table = mlx::core::zeros({1 << 27}, mlx::core::float32);  // 512 MiB
  mlx::core::eval(*large_table);
  mlx::core::synchronize();
  std::this_thread::sleep_for(std::chrono::milliseconds(3));
  large_table.reset();
  mlx::core::synchronize();
  mlx::core::clear_cache();
  std::this_thread::sleep_for(std::chrono::milliseconds(3));
  const WiredMemoryAnalysis wired_memory_analysis = analyze_wired_memory(wired_memory_sampler.finish());
  EXPECT_TRUE(wired_memory_analysis.peak_mlx_held_bytes >= (512ull << 20));
  EXPECT_TRUE(wired_memory_analysis.release_events.size() >= 1);
  // Without a system wired total there is nothing to see fall, so no lag
  // may be claimed; with one, macOS's un-wiring must be seen.
  EXPECT_TRUE(!wired_memory_analysis.release_events.empty() &&
              wired_memory_analysis.release_events.back().release_lag_microseconds.has_value() ==
                  backend_expectations::HAS_SYSTEM_WIRED_MEMORY);
}

// --- edge / boundary ---

void should_finish_at_once_when_nothing_was_released() {
  WiredMemorySampler wired_memory_sampler;
  wired_memory_sampler.start();
  const auto finish_started_at = std::chrono::steady_clock::now();
  const std::vector<WiredMemorySample> wired_memory_samples = wired_memory_sampler.finish();
  EXPECT_TRUE(std::chrono::steady_clock::now() - finish_started_at < std::chrono::milliseconds(200));
  EXPECT_TRUE(count_samples_at(wired_memory_samples, WiredSamplePoint::after_run) == 0);
  EXPECT_TRUE(wired_memory_samples.size() >= 2);
}

void should_ignore_a_second_start() {
  WiredMemorySampler wired_memory_sampler;
  wired_memory_sampler.start();
  wired_memory_sampler.start();
  const std::vector<WiredMemorySample> wired_memory_samples = wired_memory_sampler.finish();
  EXPECT_TRUE(count_samples_at(wired_memory_samples, WiredSamplePoint::run_start) == 1);
}

// --- negative ---

void should_stop_waiting_for_unwiring_at_the_limit() {
  WiredMemorySampler wired_memory_sampler(std::chrono::microseconds(1000), std::chrono::milliseconds(20));
  wired_memory_sampler.start();
  std::optional<mlx::core::array> large_table = mlx::core::zeros({1 << 27}, mlx::core::float32);
  mlx::core::eval(*large_table);
  mlx::core::synchronize();
  std::this_thread::sleep_for(std::chrono::milliseconds(3));
  large_table.reset();
  mlx::core::synchronize();
  mlx::core::clear_cache();
  std::this_thread::sleep_for(std::chrono::milliseconds(3));
  const auto finish_started_at = std::chrono::steady_clock::now();
  wired_memory_sampler.finish();
  EXPECT_TRUE(std::chrono::steady_clock::now() - finish_started_at < std::chrono::milliseconds(500));
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_sample_each_evaluation_boundary_and_the_interval_between),
      TEST_CASE(should_keep_sampling_after_the_run_until_macos_unwires_a_release),
      TEST_CASE(should_finish_at_once_when_nothing_was_released),
      TEST_CASE(should_ignore_a_second_start),
      TEST_CASE(should_stop_waiting_for_unwiring_at_the_limit),
  });
}
