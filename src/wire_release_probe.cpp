#include "wire_release_probe.h"

#include "console_output.h"
#include "progress_reporter.h"
#include "sample_statistics.h"
#include "wired_memory_sampler.h"

#include "mlx/mlx.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <thread>

namespace mlx_perf {
namespace {

// Polled finer than the run sampler: the probe does nothing else, and a
// finer poll resolves the lag more closely for the same cost.
constexpr std::chrono::microseconds PROBE_POLL_INTERVAL{200};
// MLX releases a buffer once the GPU is done with it; a wait past this
// counts as unresolved. (macOS un-wiring it, later still, has the request's
// own wait.)
constexpr double MLX_RELEASE_WAIT_SECONDS = 1.0;

// The first sample in which `has_fallen` holds, polled until the deadline.
// The probe's samples never leave it, so their sample_point labels are only
// approximate: run_start before an allocation, after_run while polling.
template <typename FallCheck>
std::optional<WiredMemorySample> poll_until(FallCheck has_fallen, double wait_seconds) {
  const WiredMemorySample first_sample = take_wired_memory_sample(WiredSamplePoint::after_run);
  for (WiredMemorySample polled_sample = first_sample;
       polled_sample.host_seconds - first_sample.host_seconds < wait_seconds;
       polled_sample = take_wired_memory_sample(WiredSamplePoint::after_run)) {
    if (has_fallen(polled_sample)) {
      return polled_sample;
    }
    std::this_thread::sleep_for(PROBE_POLL_INTERVAL);
  }
  return std::nullopt;
}

// Waits until the wired total has held within SETTLED_WIRED_BYTES for
// SETTLED_SECONDS, at most SETTLE_WAIT_SECONDS: memory the GPU used just
// before (an earlier repetition, another program) un-wires on its own timer,
// and that fall must not be read as this repetition's.
constexpr std::uint64_t SETTLED_WIRED_BYTES = 16ull << 20;
constexpr double SETTLED_SECONDS = 0.3;
constexpr double SETTLE_WAIT_SECONDS = 3.0;

void wait_for_settled_wired_memory() {
  const WiredMemorySample first_sample = take_wired_memory_sample(WiredSamplePoint::run_start);
  WiredMemorySample steady_since = first_sample;
  for (WiredMemorySample polled_sample = first_sample;
       polled_sample.host_seconds - first_sample.host_seconds < SETTLE_WAIT_SECONDS;
       polled_sample = take_wired_memory_sample(WiredSamplePoint::run_start)) {
    const std::uint64_t wired_change_bytes = polled_sample.system_wired_bytes > steady_since.system_wired_bytes
        ? polled_sample.system_wired_bytes - steady_since.system_wired_bytes
        : steady_since.system_wired_bytes - polled_sample.system_wired_bytes;
    if (wired_change_bytes > SETTLED_WIRED_BYTES) {
      steady_since = polled_sample;
    } else if (polled_sample.host_seconds - steady_since.host_seconds >= SETTLED_SECONDS) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

WiredMemorySample hold_allocation(std::uint64_t element_count) {
  std::optional<mlx::core::array> probe_allocation =
      mlx::core::zeros({static_cast<int>(element_count)}, mlx::core::float32);
  mlx::core::eval(*probe_allocation);
  mlx::core::synchronize();
  const WiredMemorySample while_held = take_wired_memory_sample(WiredSamplePoint::evaluation_boundary);
  // Released as an engine releases a finished prompt's memory: MLX returns
  // the buffer once nothing refers to it, then the cache is cleared.
  probe_allocation.reset();
  mlx::core::synchronize();
  mlx::core::clear_cache();
  return while_held;
}

WireReleaseRepetition run_repetition(std::uint64_t allocation_bytes, double unwire_wait_seconds) {
  mlx::core::clear_cache();
  mlx::core::synchronize();
  wait_for_settled_wired_memory();
  const WiredMemorySample before_allocation = take_wired_memory_sample(WiredSamplePoint::run_start);
  const WiredMemorySample while_held = hold_allocation(allocation_bytes / sizeof(float));
  WireReleaseRepetition repetition;
  repetition.wired_rise_bytes = static_cast<std::int64_t>(while_held.system_wired_bytes) -
                                static_cast<std::int64_t>(before_allocation.system_wired_bytes);
  const std::uint64_t half_allocation_bytes = allocation_bytes / 2;
  const std::optional<WiredMemorySample> mlx_released = poll_until(
      [&](const WiredMemorySample& polled_sample) {
        return mlx_held_bytes(polled_sample) + half_allocation_bytes <= mlx_held_bytes(while_held);
      },
      MLX_RELEASE_WAIT_SECONDS);
  if (!mlx_released.has_value()) {
    return repetition;
  }
  const std::optional<WiredMemorySample> unwired = poll_until(
      [&](const WiredMemorySample& polled_sample) {
        return polled_sample.system_wired_bytes + half_allocation_bytes <= while_held.system_wired_bytes;
      },
      unwire_wait_seconds);
  if (unwired.has_value()) {
    repetition.release_lag_microseconds = (unwired->host_seconds - mlx_released->host_seconds) * 1e6;
  }
  return repetition;
}

std::vector<double> resolved_lags(const WireReleaseProbeResult& probe_result) {
  std::vector<double> lag_microseconds;
  for (const WireReleaseRepetition& repetition : probe_result.repetitions) {
    if (repetition.release_lag_microseconds.has_value()) {
      lag_microseconds.push_back(*repetition.release_lag_microseconds);
    }
  }
  return lag_microseconds;
}

} // namespace

WireReleaseProbeResult measure_wire_release_lag(const WireReleaseProbeRequest& probe_request) {
  if (probe_request.allocation_bytes < sizeof(float) || probe_request.repetition_count < 1) {
    throw std::invalid_argument("wire release probe: allocation_bytes and repetition_count must be positive");
  }
  // MLX shapes are int: one array holds at most 2^31 - 1 float32 elements.
  if (probe_request.allocation_bytes / sizeof(float) > static_cast<std::uint64_t>(INT32_MAX)) {
    throw std::invalid_argument("wire release probe: allocation_bytes must be under 8 GiB");
  }
  WireReleaseProbeResult probe_result;
  probe_result.allocation_bytes = probe_request.allocation_bytes / sizeof(float) * sizeof(float);
  probe_result.wire_limit_bytes = take_wired_memory_sample(WiredSamplePoint::run_start).wire_limit_bytes;
  print_progress_line("[wire release probe] %d repetitions of %llu bytes\n", probe_request.repetition_count,
                      static_cast<unsigned long long>(probe_result.allocation_bytes));
  ProgressReporter progress_reporter("wire release repetitions", probe_request.repetition_count);
  for (int repetition_index = 0; repetition_index < probe_request.repetition_count; ++repetition_index) {
    probe_result.repetitions.push_back(
        run_repetition(probe_result.allocation_bytes, probe_request.unwire_wait_seconds));
    progress_reporter.update(repetition_index + 1);
  }
  progress_reporter.finish(probe_request.repetition_count);
  return probe_result;
}

std::optional<double> median_probe_release_lag_microseconds(const WireReleaseProbeResult& probe_result) {
  const std::vector<double> lag_microseconds = resolved_lags(probe_result);
  if (lag_microseconds.empty()) return std::nullopt;
  return summarize_samples(lag_microseconds).median;
}

std::optional<double> largest_probe_release_lag_microseconds(const WireReleaseProbeResult& probe_result) {
  const std::vector<double> lag_microseconds = resolved_lags(probe_result);
  if (lag_microseconds.empty()) return std::nullopt;
  return *std::max_element(lag_microseconds.begin(), lag_microseconds.end());
}

int count_resolved_repetitions(const WireReleaseProbeResult& probe_result) {
  return static_cast<int>(resolved_lags(probe_result).size());
}

std::int64_t median_wired_rise_bytes(const WireReleaseProbeResult& probe_result) {
  std::vector<double> wired_rises;
  for (const WireReleaseRepetition& repetition : probe_result.repetitions) {
    wired_rises.push_back(static_cast<double>(repetition.wired_rise_bytes));
  }
  return wired_rises.empty() ? 0 : static_cast<std::int64_t>(summarize_samples(wired_rises).median);
}

} // namespace mlx_perf
