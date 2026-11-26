// Unit tests for wired_memory_analysis: peaks, releases and their un-wire
// lag, inflated samples and transient jumps, on synthetic samples. No GPU.

#include "test_harness.h"
#include "wired_memory_analysis.h"
#include "wired_memory_fixture.h"

#include <vector>

namespace {

using namespace mlx_perf;
using wired_memory_fixture::GIB;
using wired_memory_fixture::LIMIT_BYTES;
using wired_memory_fixture::sample_at;

// --- positive ---

void should_find_a_release_and_time_macos_following_it() {
  const WiredMemoryAnalysis wired_memory_analysis = analyze_wired_memory(wired_memory_fixture::release_then_unwire());
  EXPECT_TRUE(wired_memory_analysis.release_events.size() == 1);
  const WireReleaseEvent& release_event = wired_memory_analysis.release_events.front();
  EXPECT_TRUE(release_event.released_sample_index == 3);
  EXPECT_TRUE(release_event.released_bytes == GIB);
  // Sample 5 still wires half a GiB and a byte: not yet "fallen by half".
  EXPECT_TRUE(release_event.unwired_sample_index == 6);
  EXPECT_NEAR(*release_event.release_lag_microseconds, 30000.0, 1e-3);
  EXPECT_NEAR(*median_release_lag_microseconds(wired_memory_analysis.release_events), 30000.0, 1e-3);
}

void should_mark_the_samples_whose_outside_figure_the_lag_inflated() {
  const WiredMemoryAnalysis wired_memory_analysis = analyze_wired_memory(wired_memory_fixture::release_then_unwire());
  EXPECT_TRUE((wired_memory_analysis.release_lag_inflated_sample_indices == std::vector<int>{3, 4, 5}));
  EXPECT_TRUE(wired_memory_analysis.largest_release_lag_inflation_bytes == GIB);
  EXPECT_TRUE(wired_memory_analysis.peak_wired_outside_mlx_bytes == static_cast<std::int64_t>(4 * GIB));
}

void should_report_peaks_and_the_least_room_under_the_limit() {
  const WiredMemoryAnalysis wired_memory_analysis = analyze_wired_memory(wired_memory_fixture::release_then_unwire());
  EXPECT_TRUE(wired_memory_analysis.sample_count == 8);
  EXPECT_TRUE(wired_memory_analysis.peak_system_wired_bytes == 4 * GIB);
  EXPECT_TRUE(wired_memory_analysis.peak_mlx_held_bytes == GIB);
  EXPECT_TRUE(wired_memory_analysis.wire_limit_bytes == LIMIT_BYTES);
  EXPECT_TRUE(*wired_memory_analysis.lowest_wire_headroom_bytes == static_cast<std::int64_t>(36 * GIB));
  EXPECT_TRUE(!wired_memory_analysis.has_crossed_wire_limit);
}

void should_find_a_jump_that_returns_within_two_samples() {
  const std::vector<WiredMemorySample> wired_memory_samples = {
      sample_at(0, 3 * GIB, 0), sample_at(1, 5 * GIB, 0), sample_at(2, 3 * GIB, 0), sample_at(3, 3 * GIB, 0)};
  EXPECT_TRUE((analyze_wired_memory(wired_memory_samples).transient_jump_sample_indices == std::vector<int>{1}));
}

// macOS un-wires memory the GPU left idle even while MLX holds it: the
// figure an engine sizes against then reads lower than others' real share.
void should_report_the_lowest_outside_figure_when_idle_mlx_memory_is_unwired() {
  WiredMemorySample idle_unwired = sample_at(10, 3 * GIB, GIB);
  idle_unwired.system_wired_bytes = 3 * GIB;
  const WiredMemoryAnalysis wired_memory_analysis = analyze_wired_memory({sample_at(0, 3 * GIB, GIB), idle_unwired});
  EXPECT_TRUE(wired_memory_analysis.lowest_wired_outside_mlx_bytes == static_cast<std::int64_t>(2 * GIB));
  EXPECT_TRUE(wired_memory_analysis.peak_wired_outside_mlx_bytes == static_cast<std::int64_t>(3 * GIB));
  EXPECT_TRUE(wired_memory_analysis.release_events.empty());
}

// --- edge / boundary ---

void should_analyze_nothing_without_samples() {
  const WiredMemoryAnalysis wired_memory_analysis = analyze_wired_memory({});
  EXPECT_TRUE(wired_memory_analysis.sample_count == 0);
  EXPECT_TRUE(wired_memory_analysis.release_events.empty());
  EXPECT_TRUE(!wired_memory_analysis.lowest_wire_headroom_bytes.has_value());
}

void should_count_a_release_exactly_at_the_minimum_and_not_one_byte_under() {
  const std::uint64_t minimum_bytes = RELEASE_EVENT_MINIMUM_BYTES;
  EXPECT_TRUE(analyze_wired_memory({sample_at(0, GIB, minimum_bytes), sample_at(1, GIB, 0)}).release_events.size() == 1);
  EXPECT_TRUE(
      analyze_wired_memory({sample_at(0, GIB, minimum_bytes - 1), sample_at(1, GIB, 0)}).release_events.empty());
}

void should_leave_a_lag_unresolved_when_macos_has_not_followed_by_the_last_sample() {
  const std::vector<WiredMemorySample> wired_memory_samples = {sample_at(0, GIB, GIB), sample_at(1, GIB, 0, GIB),
                                                               sample_at(2, GIB, 0, GIB)};
  const WiredMemoryAnalysis wired_memory_analysis = analyze_wired_memory(wired_memory_samples);
  EXPECT_TRUE(!wired_memory_analysis.release_events.front().unwired_sample_index.has_value());
  EXPECT_TRUE(!median_release_lag_microseconds(wired_memory_analysis.release_events).has_value());
  EXPECT_TRUE((wired_memory_analysis.release_lag_inflated_sample_indices == std::vector<int>{1, 2}));
}

void should_resolve_a_lag_of_zero_when_macos_follows_in_the_same_sample() {
  const WiredMemoryAnalysis wired_memory_analysis =
      analyze_wired_memory({sample_at(0, GIB, GIB), sample_at(1, GIB, 0)});
  EXPECT_NEAR(*wired_memory_analysis.release_events.front().release_lag_microseconds, 0.0, 1e-9);
  EXPECT_TRUE(wired_memory_analysis.release_lag_inflated_sample_indices.empty());
}

// --- negative ---

void should_flag_crossing_the_limit_and_skip_headroom_when_the_limit_is_unknown() {
  EXPECT_TRUE(analyze_wired_memory({sample_at(0, 41 * GIB, 0)}).has_crossed_wire_limit);
  const WiredMemoryAnalysis unknown_limit = analyze_wired_memory({sample_at(0, 41 * GIB, 0, 0, 0)});
  EXPECT_TRUE(!unknown_limit.has_crossed_wire_limit);
  EXPECT_TRUE(!unknown_limit.lowest_wire_headroom_bytes.has_value());
  EXPECT_TRUE(unknown_limit.wire_limit_bytes == 0);
}

void should_not_call_a_lasting_rise_transient() {
  const std::vector<WiredMemorySample> wired_memory_samples = {
      sample_at(0, 3 * GIB, 0), sample_at(1, 5 * GIB, 0), sample_at(2, 5 * GIB, 0), sample_at(3, 5 * GIB, 0)};
  EXPECT_TRUE(analyze_wired_memory(wired_memory_samples).transient_jump_sample_indices.empty());
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_find_a_release_and_time_macos_following_it),
      TEST_CASE(should_mark_the_samples_whose_outside_figure_the_lag_inflated),
      TEST_CASE(should_report_peaks_and_the_least_room_under_the_limit),
      TEST_CASE(should_find_a_jump_that_returns_within_two_samples),
      TEST_CASE(should_report_the_lowest_outside_figure_when_idle_mlx_memory_is_unwired),
      TEST_CASE(should_analyze_nothing_without_samples),
      TEST_CASE(should_count_a_release_exactly_at_the_minimum_and_not_one_byte_under),
      TEST_CASE(should_leave_a_lag_unresolved_when_macos_has_not_followed_by_the_last_sample),
      TEST_CASE(should_resolve_a_lag_of_zero_when_macos_follows_in_the_same_sample),
      TEST_CASE(should_flag_crossing_the_limit_and_skip_headroom_when_the_limit_is_unknown),
      TEST_CASE(should_not_call_a_lasting_rise_transient),
  });
}
