// Unit tests for recording_slots: which recording a hook finds, on which
// thread, for each reach. No GPU: the recordings are plain integers.

#include "recording_slots.h"
#include "test_harness.h"

#include <thread>

namespace {

using mlx_perf::RecordingReach;

// Each test case uses its own state type, so each gets its own slots and no
// test sees a recording another left open.
template <int TestNumber>
struct TestRecording {
  int recorded_event_count = 0;
};

template <typename RecordingState>
RecordingState* recording_seen_on_another_thread() {
  RecordingState* seen_recording = nullptr;
  std::thread other_thread([&seen_recording]() {
    seen_recording = typename mlx_perf::RecordingSlots<RecordingState>::ActiveRecording().recording_state();
  });
  other_thread.join();
  return seen_recording;
}

// --- positive ---

void should_show_a_whole_process_recording_on_every_thread() {
  using Slots = mlx_perf::RecordingSlots<TestRecording<1>>;
  TestRecording<1> process_recording;
  Slots::swap_open_recording(RecordingReach::whole_process, &process_recording);
  EXPECT_TRUE(Slots::ActiveRecording().recording_state() == &process_recording);
  EXPECT_TRUE(recording_seen_on_another_thread<TestRecording<1>>() == &process_recording);
  Slots::swap_open_recording(RecordingReach::whole_process, nullptr);
}

void should_keep_a_calling_thread_recording_to_its_thread() {
  using Slots = mlx_perf::RecordingSlots<TestRecording<2>>;
  TestRecording<2> thread_recording;
  Slots::swap_open_recording(RecordingReach::calling_thread, &thread_recording);
  EXPECT_TRUE(Slots::ActiveRecording().recording_state() == &thread_recording);
  EXPECT_TRUE(recording_seen_on_another_thread<TestRecording<2>>() == nullptr);
  Slots::swap_open_recording(RecordingReach::calling_thread, nullptr);
}

// --- edge / boundary ---

void should_prefer_the_threads_own_recording_and_return_what_each_swap_replaced() {
  using Slots = mlx_perf::RecordingSlots<TestRecording<3>>;
  TestRecording<3> process_recording;
  TestRecording<3> thread_recording;
  EXPECT_TRUE(Slots::swap_open_recording(RecordingReach::whole_process, &process_recording) == nullptr);
  EXPECT_TRUE(Slots::swap_open_recording(RecordingReach::calling_thread, &thread_recording) == nullptr);
  EXPECT_TRUE(Slots::ActiveRecording().recording_state() == &thread_recording);
  EXPECT_TRUE(Slots::swap_open_recording(RecordingReach::calling_thread, nullptr) == &thread_recording);
  EXPECT_TRUE(Slots::ActiveRecording().recording_state() == &process_recording);
  EXPECT_TRUE(Slots::swap_open_recording(RecordingReach::whole_process, nullptr) == &process_recording);
}

// --- negative ---

void should_find_nothing_when_no_recording_is_open() {
  using Slots = mlx_perf::RecordingSlots<TestRecording<4>>;
  EXPECT_TRUE(Slots::ActiveRecording().recording_state() == nullptr);
  EXPECT_TRUE(recording_seen_on_another_thread<TestRecording<4>>() == nullptr);
}

void should_find_nothing_while_a_recording_is_paused() {
  using Slots = mlx_perf::RecordingSlots<TestRecording<5>>;
  TestRecording<5> process_recording;
  Slots::swap_open_recording(RecordingReach::whole_process, &process_recording);
  TestRecording<5>* paused_recording = Slots::swap_open_recording(RecordingReach::whole_process, nullptr);
  EXPECT_TRUE(recording_seen_on_another_thread<TestRecording<5>>() == nullptr);
  Slots::swap_open_recording(RecordingReach::whole_process, paused_recording);
  EXPECT_TRUE(recording_seen_on_another_thread<TestRecording<5>>() == &process_recording);
  Slots::swap_open_recording(RecordingReach::whole_process, nullptr);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_show_a_whole_process_recording_on_every_thread),
      TEST_CASE(should_keep_a_calling_thread_recording_to_its_thread),
      TEST_CASE(should_prefer_the_threads_own_recording_and_return_what_each_swap_replaced),
      TEST_CASE(should_find_nothing_when_no_recording_is_open),
      TEST_CASE(should_find_nothing_while_a_recording_is_paused),
  });
}
