// Unit tests for live_view_store: versioning, sample cap, and the run
// request hand-off between threads.

#include "live_view_store.h"
#include "test_harness.h"

#include <chrono>
#include <thread>

namespace {

using namespace mlx_plan;
using std::chrono::milliseconds;

// --- positive ---

void should_bump_both_versions_when_views_are_published() {
  LiveViewStore live_view_store;
  live_view_store.publish_views({"job", "stages", "tasks", "memory"});
  EXPECT_TRUE(live_view_store.version() == 1);
  EXPECT_TRUE(live_view_store.views_version() == 1);
  EXPECT_TRUE(live_view_store.views().tasks_html == "tasks");
}

void should_bump_only_the_version_when_status_changes() {
  LiveViewStore live_view_store;
  live_view_store.set_status("measuring");
  EXPECT_TRUE(live_view_store.version() == 1);
  EXPECT_TRUE(live_view_store.views_version() == 0);
  EXPECT_TRUE(live_view_store.status() == "measuring");
}

void should_wake_a_waiter_when_another_thread_publishes() {
  LiveViewStore live_view_store;
  std::thread publisher_thread(&LiveViewStore::set_status, &live_view_store,
                               std::string("done"));
  const std::uint64_t woken_version =
      live_view_store.wait_for_version_after(0, milliseconds(2000));
  publisher_thread.join();
  EXPECT_TRUE(woken_version >= 1);
}

void should_hand_a_run_request_to_the_main_thread_once() {
  LiveViewStore live_view_store;
  live_view_store.request_run();
  live_view_store.request_run();
  EXPECT_TRUE(live_view_store.wait_for_run_request(milliseconds(10)));
  // Two requests while one was pending collapse into a single run.
  EXPECT_TRUE(!live_view_store.wait_for_run_request(milliseconds(10)));
}

// --- edge / boundary ---

void should_keep_only_the_newest_six_hundred_memory_samples() {
  LiveViewStore live_view_store;
  for (int sample_index = 0; sample_index < 650; ++sample_index) {
    live_view_store.add_memory_sample({static_cast<double>(sample_index), 1, 2, 3});
  }
  const auto memory_samples = live_view_store.memory_samples();
  EXPECT_TRUE(memory_samples.size() == 600);
  EXPECT_TRUE(memory_samples.front().seconds_since_start == 50.0);
  EXPECT_TRUE(live_view_store.version() == 0);
}

// --- negative ---

void should_return_unchanged_version_when_wait_times_out() {
  LiveViewStore live_view_store;
  live_view_store.set_status("idle");
  EXPECT_TRUE(live_view_store.wait_for_version_after(1, milliseconds(20)) == 1);
}

void should_report_no_run_when_none_was_requested() {
  LiveViewStore live_view_store;
  EXPECT_TRUE(!live_view_store.wait_for_run_request(milliseconds(20)));
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_bump_both_versions_when_views_are_published),
      TEST_CASE(should_bump_only_the_version_when_status_changes),
      TEST_CASE(should_wake_a_waiter_when_another_thread_publishes),
      TEST_CASE(should_hand_a_run_request_to_the_main_thread_once),
      TEST_CASE(should_keep_only_the_newest_six_hundred_memory_samples),
      TEST_CASE(should_return_unchanged_version_when_wait_times_out),
      TEST_CASE(should_report_no_run_when_none_was_requested),
  });
}
