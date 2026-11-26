#pragma once

// Wired memory sampler: one concern only — take wired memory samples during a
// measured run: at its start and end, at every evaluation boundary (as an
// EvaluationBoundaryListener), on a fine interval in between, and after the
// run until macOS has un-wired what MLX released in it.
//
// Why a fine interval as well as boundaries: an engine reads its state at
// evaluation boundaries, so those samples show what the engine saw; the
// interval samples show what happened between them, which is where a release
// and its un-wiring sit. Why sample after the run: macOS un-wires released
// memory a fraction of a second later, often after a short run has ended, and
// a release lag needs its end to be measured.
//
// Cost: each sample is two cheap system calls and two reads of MLX's
// allocator counters (which take its lock). The interval thread competes with
// the measured run for that lock, so sample only runs whose timing is not
// judged (procedures: the observed run).

#include "evaluation_boundary_listener.h"
#include "wired_memory_sample.h"

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

namespace mlx_perf {

// One sample now: system wired memory, the wire limit and MLX's holding.
WiredMemorySample take_wired_memory_sample(WiredSamplePoint sample_point);

inline constexpr std::chrono::microseconds DEFAULT_WIRED_SAMPLE_INTERVAL{1000};
// How long to keep sampling after a run for macOS to follow MLX's releases.
inline constexpr std::chrono::milliseconds DEFAULT_UNWIRE_WAIT_LIMIT{3000};

class WiredMemorySampler final : public EvaluationBoundaryListener {
 public:
  explicit WiredMemorySampler(std::chrono::microseconds sample_interval = DEFAULT_WIRED_SAMPLE_INTERVAL,
                              std::chrono::milliseconds unwire_wait_limit = DEFAULT_UNWIRE_WAIT_LIMIT);
  ~WiredMemorySampler() override;
  WiredMemorySampler(const WiredMemorySampler&) = delete;
  WiredMemorySampler& operator=(const WiredMemorySampler&) = delete;

  // Takes the run_start sample and starts the interval thread. Calling it
  // again while sampling does nothing.
  void start();
  void on_evaluation_boundary() override;
  // Stops the interval thread, takes the run_end sample, then samples at the
  // interval until every release in the run is un-wired or the wait limit
  // passes. Returns every sample in time order. Call once, after start().
  std::vector<WiredMemorySample> finish();

 private:
  void sample_on_interval();
  void add_sample(const WiredMemorySample& wired_memory_sample);
  void sample_until_unwired();

  std::chrono::microseconds sample_interval_;
  std::chrono::milliseconds unwire_wait_limit_;
  std::mutex samples_mutex_;
  std::vector<WiredMemorySample> wired_memory_samples_;
  std::atomic<bool> should_keep_sampling_{false};
  std::thread interval_thread_;
};

} // namespace mlx_perf
