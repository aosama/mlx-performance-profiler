// Unit tests for cuda_host_clock: it must read the clock the passes anchor
// on (steady_clock), which is the one cupti_session hands CUPTI; that CUPTI's
// records land on it is checked in cupti_session_test.

#include "gpu_host_clock.h"
#include "test_harness.h"

#include <chrono>
#include <thread>

namespace {

using mlx_perf::current_gpu_host_seconds;

// --- positive ---

void should_read_the_same_clock_as_steady_clock() {
  // The passes anchor on std::chrono::steady_clock; the probe relies on the
  // two being one clock (CLOCK_MONOTONIC), so they must agree closely.
  const double steady_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  const double gpu_host_seconds = current_gpu_host_seconds();
  constexpr double SAME_CLOCK_TOLERANCE_SECONDS = 0.001;
  EXPECT_NEAR(gpu_host_seconds, steady_seconds, SAME_CLOCK_TOLERANCE_SECONDS);
}

void should_advance_with_wall_time() {
  const double started_seconds = current_gpu_host_seconds();
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  const double elapsed_seconds = current_gpu_host_seconds() - started_seconds;
  EXPECT_TRUE(elapsed_seconds >= 0.015 && elapsed_seconds < 1.0);
}

// --- edge / boundary ---

void should_never_run_backwards_between_immediate_reads() {
  double previous_seconds = current_gpu_host_seconds();
  for (int read_index = 0; read_index < 1000; ++read_index) {
    const double current_seconds = current_gpu_host_seconds();
    EXPECT_TRUE(current_seconds >= previous_seconds);
    previous_seconds = current_seconds;
  }
}

// --- negative ---

void should_not_read_zero_or_a_negative_time() {
  // Zero is what an unmeasured window holds: a clock reading of zero would
  // be indistinguishable from it.
  EXPECT_TRUE(current_gpu_host_seconds() > 0.0);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_read_the_same_clock_as_steady_clock),
      TEST_CASE(should_advance_with_wall_time),
      TEST_CASE(should_never_run_backwards_between_immediate_reads),
      TEST_CASE(should_not_read_zero_or_a_negative_time),
  });
}
