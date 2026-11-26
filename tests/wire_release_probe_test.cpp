// Integration tests for wire_release_probe on the Metal GPU: the probe sees
// the system's wired total rise with an allocation and fall after MLX
// releases it. Checked for presence and plausibility, never exact lags.

#include "test_harness.h"
#include "wire_release_probe.h"

#include <stdexcept>

namespace {

using namespace mlx_perf;

constexpr std::uint64_t PROBE_BYTES = 256ull << 20;

bool is_refused(const WireReleaseProbeRequest& probe_request) {
  try {
    measure_wire_release_lag(probe_request);
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

// --- positive ---

void should_see_an_allocation_wired_then_unwired_after_a_lag() {
  const WireReleaseProbeResult probe_result = measure_wire_release_lag({PROBE_BYTES, 2, 5.0});
  EXPECT_TRUE(probe_result.repetitions.size() == 2);
  EXPECT_TRUE(probe_result.allocation_bytes == PROBE_BYTES);
  EXPECT_TRUE(probe_result.wire_limit_bytes > 0);
  // The wired total rises by about the allocation; other activity on the
  // machine moves it a little either way.
  EXPECT_TRUE(median_wired_rise_bytes(probe_result) > static_cast<std::int64_t>(PROBE_BYTES / 2));
  EXPECT_TRUE(count_resolved_repetitions(probe_result) == 2);
  EXPECT_TRUE(*median_probe_release_lag_microseconds(probe_result) >= 0.0);
  EXPECT_TRUE(*largest_probe_release_lag_microseconds(probe_result) >=
              *median_probe_release_lag_microseconds(probe_result));
}

// --- edge / boundary ---

void should_round_the_allocation_down_to_whole_float32_elements() {
  const WireReleaseProbeResult probe_result = measure_wire_release_lag({PROBE_BYTES + 3, 1, 5.0});
  EXPECT_TRUE(probe_result.allocation_bytes == PROBE_BYTES);
}

void should_report_an_unresolved_lag_when_the_wait_is_too_short_to_see_it() {
  const WireReleaseProbeResult probe_result = measure_wire_release_lag({PROBE_BYTES, 1, 0.0});
  EXPECT_TRUE(count_resolved_repetitions(probe_result) <= 1);
  if (count_resolved_repetitions(probe_result) == 0) {
    EXPECT_TRUE(!median_probe_release_lag_microseconds(probe_result).has_value());
  }
}

// --- negative ---

void should_refuse_an_empty_allocation_or_no_repetitions() {
  EXPECT_TRUE(is_refused({0, 1, 1.0}));
  EXPECT_TRUE(is_refused({3, 1, 1.0}));
  EXPECT_TRUE(is_refused({PROBE_BYTES, 0, 1.0}));
}

void should_refuse_an_allocation_one_mlx_array_cannot_hold() {
  EXPECT_TRUE(is_refused({8ull << 30, 1, 1.0}));
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_see_an_allocation_wired_then_unwired_after_a_lag),
      TEST_CASE(should_round_the_allocation_down_to_whole_float32_elements),
      TEST_CASE(should_report_an_unresolved_lag_when_the_wait_is_too_short_to_see_it),
      TEST_CASE(should_refuse_an_empty_allocation_or_no_repetitions),
      TEST_CASE(should_refuse_an_allocation_one_mlx_array_cannot_hold),
  });
}
