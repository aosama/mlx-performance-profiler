#pragma once

// Live view store: one concern only — the hand-off between the main thread
// (which owns all MLX work) and the HTTP threads (which must never touch
// MLX: its default streams and Metal encoders are per-thread). Everything
// crossing threads is plain data: rendered HTML strings, memory samples, a
// status line, and run requests. One mutex, two condition variables.

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace mlx_plan {

struct RenderedViews {
  std::string job_html;
  std::string stages_html;
  std::string tasks_html;
  std::string memory_html;  // per-run part; the live chart is drawn per request
};

struct MemorySample {
  double seconds_since_start = 0.0;
  std::size_t active_bytes = 0;
  std::size_t peak_bytes = 0;
  std::size_t cache_bytes = 0;
};

class LiveViewStore {
 public:
  // Publishing views or status bumps the version that event streams follow.
  void publish_views(RenderedViews rendered_views);
  void set_status(std::string status_line);
  void add_memory_sample(const MemorySample& memory_sample);

  RenderedViews views() const;
  std::string status() const;
  std::uint64_t version() const;
  // Bumped only by publish_views: lets a page skip reloading a tab (and
  // losing its sort order) when merely the status line changed.
  std::uint64_t views_version() const;
  std::vector<MemorySample> memory_samples() const;

  // Blocks until the version passes `seen_version` or `timeout` elapses;
  // returns the current version either way.
  std::uint64_t wait_for_version_after(std::uint64_t seen_version,
                                       std::chrono::milliseconds timeout) const;

  // HTTP thread → main thread. Repeated requests while one is pending
  // collapse into one run.
  void request_run();
  // Main thread: waits up to `timeout`; true (and consumed) when requested.
  bool wait_for_run_request(std::chrono::milliseconds timeout);

 private:
  static constexpr std::size_t MAX_MEMORY_SAMPLES = 600;  // 10 min at 1 Hz

  mutable std::mutex store_mutex_;
  mutable std::condition_variable version_changed_;
  std::condition_variable run_requested_;
  RenderedViews rendered_views_;
  std::string status_line_ = "starting";
  std::uint64_t version_ = 0;
  std::uint64_t views_version_ = 0;
  std::deque<MemorySample> memory_samples_;
  bool is_run_requested_ = false;
};

} // namespace mlx_plan
