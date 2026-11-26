// Unit tests for c_api_support: the last error, exceptions turned into C
// statuses, string checks and metric copying. No GPU.

#include "c_api_support.h"
#include "test_harness.h"

#include <stdexcept>
#include <string>
#include <thread>

namespace {

using namespace mlx_perf;

int status_after_throwing(void (*throw_failure)()) {
  try {
    throw_failure();
  } catch (...) {
    return record_failure_in_flight();
  }
  return 0;
}

void throw_invalid_argument() { throw std::invalid_argument("bad name"); }
void throw_integer() { throw 42; }

// --- positive ---

void should_record_the_message_of_the_exception_in_flight() {
  clear_last_c_api_error();
  EXPECT_TRUE(status_after_throwing(throw_invalid_argument) == 1);
  EXPECT_TRUE(std::string(last_c_api_error()) == "bad name");
  clear_last_c_api_error();
  EXPECT_TRUE(std::string(last_c_api_error()).empty());
}

void should_copy_every_field_and_map_each_verdict() {
  PairedComparison paired_comparison;
  paired_comparison.baseline.median = 400.0;
  paired_comparison.candidate.minimum = 150.0;
  paired_comparison.median_difference = -250.0;
  paired_comparison.difference_interval_coverage = 0.961;
  paired_comparison.rounds_candidate_faster = 12;
  paired_comparison.round_count = 12;
  paired_comparison.verdict = ComparisonVerdict::candidate_faster;
  mlx_perf_metric_comparison metric_comparison{};
  fill_metric_comparison(paired_comparison, metric_comparison);
  EXPECT_TRUE(metric_comparison.baseline_median_us == 400.0);
  EXPECT_TRUE(metric_comparison.candidate_min_us == 150.0);
  EXPECT_TRUE(metric_comparison.median_difference_us == -250.0);
  EXPECT_TRUE(metric_comparison.difference_interval_coverage == 0.961);
  EXPECT_TRUE(metric_comparison.verdict == MLX_PERF_VERDICT_CANDIDATE_FASTER);
  paired_comparison.verdict = ComparisonVerdict::candidate_slower;
  fill_metric_comparison(paired_comparison, metric_comparison);
  EXPECT_TRUE(metric_comparison.verdict == MLX_PERF_VERDICT_CANDIDATE_SLOWER);
  paired_comparison.verdict = ComparisonVerdict::within_spread;
  fill_metric_comparison(paired_comparison, metric_comparison);
  EXPECT_TRUE(metric_comparison.verdict == MLX_PERF_VERDICT_WITHIN_SPREAD);
}

void should_copy_a_memory_comparison_in_bytes() {
  PairedComparison paired_comparison;
  paired_comparison.baseline.median = 64.0 * 1048576.0;
  paired_comparison.candidate.median = 320.0 * 1048576.0;
  paired_comparison.median_difference = 256.0 * 1048576.0;
  paired_comparison.difference_interval_low = 250.0 * 1048576.0;
  paired_comparison.difference_interval_high = 260.0 * 1048576.0;
  paired_comparison.relative_difference = 4.0;
  paired_comparison.verdict = ComparisonVerdict::candidate_slower;
  paired_comparison.rounds_candidate_faster = 0;
  paired_comparison.round_count = 12;
  mlx_perf_memory_comparison memory_comparison{};
  fill_memory_comparison(paired_comparison, memory_comparison);
  EXPECT_TRUE(memory_comparison.baseline_median_bytes == 64.0 * 1048576.0);
  EXPECT_TRUE(memory_comparison.median_difference_bytes == 256.0 * 1048576.0);
  EXPECT_TRUE(memory_comparison.difference_interval_low_bytes == 250.0 * 1048576.0);
  EXPECT_TRUE(memory_comparison.relative_difference == 4.0);
  EXPECT_TRUE(memory_comparison.verdict == MLX_PERF_VERDICT_CANDIDATE_SLOWER);
  EXPECT_TRUE(memory_comparison.rounds_candidate_less == 0);
  EXPECT_TRUE(memory_comparison.round_count == 12);
}

// --- edge / boundary ---

void should_keep_each_threads_last_error_its_own() {
  clear_last_c_api_error();
  std::thread failing_thread([] { status_after_throwing(throw_invalid_argument); });
  failing_thread.join();
  EXPECT_TRUE(std::string(last_c_api_error()).empty());
}

void should_accept_an_empty_list_of_strings() {
  bool has_thrown = false;
  try {
    require_strings({}, "entry");
  } catch (const std::invalid_argument&) {
    has_thrown = true;
  }
  EXPECT_TRUE(!has_thrown);
}

// --- negative ---

void should_name_the_entry_point_for_a_null_string() {
  std::string failure_message;
  try {
    require_strings({"present", nullptr}, "mlx_perf_example_entry");
  } catch (const std::invalid_argument& failure) {
    failure_message = failure.what();
  }
  EXPECT_TRUE(failure_message == "mlx_perf_example_entry: null argument");
}

void should_record_an_unknown_failure_for_a_non_standard_exception() {
  EXPECT_TRUE(status_after_throwing(throw_integer) == 1);
  EXPECT_TRUE(std::string(last_c_api_error()) == "unknown failure");
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_record_the_message_of_the_exception_in_flight),
      TEST_CASE(should_copy_every_field_and_map_each_verdict),
      TEST_CASE(should_copy_a_memory_comparison_in_bytes),
      TEST_CASE(should_keep_each_threads_last_error_its_own),
      TEST_CASE(should_accept_an_empty_list_of_strings),
      TEST_CASE(should_name_the_entry_point_for_a_null_string),
      TEST_CASE(should_record_an_unknown_failure_for_a_non_standard_exception),
  });
}
