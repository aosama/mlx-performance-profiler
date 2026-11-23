#include "live_view_store.h"

#include <utility>

namespace mlx_plan {

void LiveViewStore::publish_views(RenderedViews rendered_views) {
  {
    std::lock_guard<std::mutex> store_lock(store_mutex_);
    rendered_views_ = std::move(rendered_views);
    ++version_;
    ++views_version_;
  }
  version_changed_.notify_all();
}

void LiveViewStore::set_status(std::string status_line) {
  {
    std::lock_guard<std::mutex> store_lock(store_mutex_);
    status_line_ = std::move(status_line);
    ++version_;
  }
  version_changed_.notify_all();
}

// Samples do not bump the version: at 1 Hz they would turn every event
// stream into a firehose; the memory tab polls its chart instead.
void LiveViewStore::add_memory_sample(const MemorySample& memory_sample) {
  std::lock_guard<std::mutex> store_lock(store_mutex_);
  memory_samples_.push_back(memory_sample);
  if (memory_samples_.size() > MAX_MEMORY_SAMPLES) {
    memory_samples_.pop_front();
  }
}

RenderedViews LiveViewStore::views() const {
  std::lock_guard<std::mutex> store_lock(store_mutex_);
  return rendered_views_;
}

std::string LiveViewStore::status() const {
  std::lock_guard<std::mutex> store_lock(store_mutex_);
  return status_line_;
}

std::uint64_t LiveViewStore::version() const {
  std::lock_guard<std::mutex> store_lock(store_mutex_);
  return version_;
}

std::uint64_t LiveViewStore::views_version() const {
  std::lock_guard<std::mutex> store_lock(store_mutex_);
  return views_version_;
}

std::vector<MemorySample> LiveViewStore::memory_samples() const {
  std::lock_guard<std::mutex> store_lock(store_mutex_);
  return {memory_samples_.begin(), memory_samples_.end()};
}

std::uint64_t LiveViewStore::wait_for_version_after(
    std::uint64_t seen_version, std::chrono::milliseconds timeout) const {
  std::unique_lock<std::mutex> store_lock(store_mutex_);
  version_changed_.wait_for(store_lock, timeout,
                            [this, seen_version] { return version_ > seen_version; });
  return version_;
}

void LiveViewStore::request_run() {
  {
    std::lock_guard<std::mutex> store_lock(store_mutex_);
    is_run_requested_ = true;
  }
  run_requested_.notify_one();
}

bool LiveViewStore::wait_for_run_request(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> store_lock(store_mutex_);
  run_requested_.wait_for(store_lock, timeout, [this] { return is_run_requested_; });
  const bool was_requested = is_run_requested_;
  is_run_requested_ = false;
  return was_requested;
}

} // namespace mlx_plan
