// Unit tests for progress_reporter's ETA estimate (the printing itself is a
// thin printf wrapper and is exercised by every demo run).

#include "progress_reporter.h"
#include "test_harness.h"

namespace {

using mlx_plan::estimate_remaining_seconds;

constexpr double EXACT_SECONDS = 1e-9;

// --- positive ---

void should_extrapolate_remaining_time_from_observed_rate() {
  // 10 s for 25 of 100 units → 0.4 s per unit → 75 units left = 30 s.
  EXPECT_NEAR(estimate_remaining_seconds(10.0, 25, 100), 30.0, EXACT_SECONDS);
}

// --- edge / boundary ---

void should_report_zero_eta_before_any_unit_completes() {
  EXPECT_NEAR(estimate_remaining_seconds(3.0, 0, 100), 0.0, EXACT_SECONDS);
}

void should_report_zero_eta_when_all_units_complete() {
  EXPECT_NEAR(estimate_remaining_seconds(3.0, 100, 100), 0.0, EXACT_SECONDS);
}

void should_report_zero_eta_for_empty_workload() {
  EXPECT_NEAR(estimate_remaining_seconds(0.0, 0, 0), 0.0, EXACT_SECONDS);
}

// --- negative ---

void should_never_report_negative_eta_when_completed_exceeds_total() {
  EXPECT_NEAR(estimate_remaining_seconds(5.0, 120, 100), 0.0, EXACT_SECONDS);
}

void should_never_report_negative_eta_for_negative_completed_count() {
  EXPECT_NEAR(estimate_remaining_seconds(5.0, -3, 100), 0.0, EXACT_SECONDS);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_extrapolate_remaining_time_from_observed_rate),
      TEST_CASE(should_report_zero_eta_before_any_unit_completes),
      TEST_CASE(should_report_zero_eta_when_all_units_complete),
      TEST_CASE(should_report_zero_eta_for_empty_workload),
      TEST_CASE(should_never_report_negative_eta_when_completed_exceeds_total),
      TEST_CASE(should_never_report_negative_eta_for_negative_completed_count),
  });
}
