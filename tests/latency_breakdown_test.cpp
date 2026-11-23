// Unit tests for latency_breakdown: the Spark StagePage decomposition over
// synthetic timestamps. No GPU involved.

#include "latency_breakdown.h"
#include "test_harness.h"

namespace {

using mlx_plan::decompose_node_latency;
using mlx_plan::LatencyBreakdown;
using mlx_plan::NodeClass;
using mlx_plan::NodeTimestamps;

constexpr double EXACT_MICROSECONDS = 1e-9;

NodeTimestamps make_timestamps(
    double issue_started,
    double issue_finished,
    double stream_ready,
    double cross_stream_inputs_ready,
    double completed) {
  NodeTimestamps timestamps;
  timestamps.issue_started_microseconds = issue_started;
  timestamps.issue_finished_microseconds = issue_finished;
  timestamps.stream_ready_microseconds = stream_ready;
  timestamps.cross_stream_inputs_ready_microseconds = cross_stream_inputs_ready;
  timestamps.completed_microseconds = completed;
  return timestamps;
}

NodeTimestamps with_gpu_window(
    NodeTimestamps timestamps, double gpu_started, double gpu_finished) {
  timestamps.has_gpu_window = true;
  timestamps.gpu_started_microseconds = gpu_started;
  timestamps.gpu_finished_microseconds = gpu_finished;
  return timestamps;
}

double sum_of_slices(const LatencyBreakdown& breakdown) {
  return breakdown.wait_on_cpu_microseconds +
      breakdown.wait_on_gpu_microseconds +
      breakdown.wait_on_dependency_microseconds +
      breakdown.scheduler_delay_microseconds +
      breakdown.compute_microseconds +
      breakdown.completion_latency_microseconds;
}

// --- positive ---

void should_attribute_residual_to_compute_when_nothing_blocks() {
  const LatencyBreakdown breakdown =
      decompose_node_latency(make_timestamps(0, 1, 0, 0, 11));
  EXPECT_NEAR(breakdown.wait_on_cpu_microseconds, 1, EXACT_MICROSECONDS);
  EXPECT_NEAR(breakdown.wait_on_gpu_microseconds, 0, EXACT_MICROSECONDS);
  EXPECT_NEAR(breakdown.wait_on_dependency_microseconds, 0, EXACT_MICROSECONDS);
  EXPECT_NEAR(breakdown.compute_microseconds, 10, EXACT_MICROSECONDS);
  EXPECT_TRUE(breakdown.classification == NodeClass::compute);
}

void should_attribute_cross_stream_wait_to_dependency_at_a_join() {
  const LatencyBreakdown breakdown =
      decompose_node_latency(make_timestamps(0, 1, 2, 10, 12));
  EXPECT_NEAR(breakdown.wait_on_cpu_microseconds, 1, EXACT_MICROSECONDS);
  EXPECT_NEAR(breakdown.wait_on_gpu_microseconds, 1, EXACT_MICROSECONDS);
  EXPECT_NEAR(breakdown.wait_on_dependency_microseconds, 8, EXACT_MICROSECONDS);
  EXPECT_NEAR(breakdown.compute_microseconds, 2, EXACT_MICROSECONDS);
  EXPECT_TRUE(breakdown.classification == NodeClass::wait_on_dependency);
}

void should_attribute_stream_queueing_to_wait_on_gpu() {
  const LatencyBreakdown breakdown =
      decompose_node_latency(make_timestamps(0, 1, 9, 0, 10));
  EXPECT_NEAR(breakdown.wait_on_gpu_microseconds, 8, EXACT_MICROSECONDS);
  EXPECT_TRUE(breakdown.classification == NodeClass::wait_on_gpu);
}

void should_classify_host_bound_node_as_wait_on_cpu() {
  const LatencyBreakdown breakdown =
      decompose_node_latency(make_timestamps(5, 25, 0, 0, 26));
  EXPECT_NEAR(breakdown.wait_on_cpu_microseconds, 20, EXACT_MICROSECONDS);
  EXPECT_NEAR(breakdown.wall_microseconds, 21, EXACT_MICROSECONDS);
  EXPECT_TRUE(breakdown.classification == NodeClass::wait_on_cpu);
}

void should_make_slices_sum_to_wall() {
  const NodeTimestamps scenarios[] = {
      make_timestamps(0, 3, 7, 12, 30),
      make_timestamps(2, 4, 1, 3, 9),
      make_timestamps(10, 11, 40, 20, 41)};
  for (const NodeTimestamps& scenario : scenarios) {
    const LatencyBreakdown breakdown = decompose_node_latency(scenario);
    EXPECT_NEAR(
        sum_of_slices(breakdown), breakdown.wall_microseconds,
        EXACT_MICROSECONDS);
  }
}

// --- edge / boundary ---

void should_classify_tie_between_waiting_and_compute_as_compute() {
  const LatencyBreakdown breakdown =
      decompose_node_latency(make_timestamps(0, 5, 0, 0, 10));
  EXPECT_NEAR(breakdown.wait_on_cpu_microseconds, 5, EXACT_MICROSECONDS);
  EXPECT_NEAR(breakdown.compute_microseconds, 5, EXACT_MICROSECONDS);
  EXPECT_TRUE(breakdown.classification == NodeClass::compute);
}

void should_not_charge_dependency_wait_when_remote_data_arrived_first() {
  const LatencyBreakdown breakdown =
      decompose_node_latency(make_timestamps(0, 1, 8, 4, 10));
  EXPECT_NEAR(breakdown.wait_on_dependency_microseconds, 0, EXACT_MICROSECONDS);
  EXPECT_NEAR(breakdown.wait_on_gpu_microseconds, 7, EXACT_MICROSECONDS);
}

void should_yield_all_zero_slices_for_zero_length_window() {
  const LatencyBreakdown breakdown =
      decompose_node_latency(make_timestamps(0, 0, 0, 0, 0));
  EXPECT_NEAR(breakdown.wall_microseconds, 0, EXACT_MICROSECONDS);
  EXPECT_NEAR(sum_of_slices(breakdown), 0, EXACT_MICROSECONDS);
  EXPECT_TRUE(breakdown.classification == NodeClass::compute);
}

// --- negative (inconsistent observations must not produce nonsense) ---

void should_clamp_ready_instant_after_completion_without_negative_compute() {
  const LatencyBreakdown breakdown =
      decompose_node_latency(make_timestamps(0, 1, 0, 50, 10));
  EXPECT_NEAR(breakdown.wait_on_dependency_microseconds, 9, EXACT_MICROSECONDS);
  EXPECT_NEAR(breakdown.compute_microseconds, 0, EXACT_MICROSECONDS);
  EXPECT_NEAR(sum_of_slices(breakdown), 10, EXACT_MICROSECONDS);
}

void should_report_zero_wall_when_completion_precedes_issue() {
  const LatencyBreakdown breakdown =
      decompose_node_latency(make_timestamps(20, 25, 0, 0, 5));
  EXPECT_NEAR(breakdown.wall_microseconds, 0, EXACT_MICROSECONDS);
  EXPECT_TRUE(breakdown.wait_on_cpu_microseconds >= 0);
  EXPECT_TRUE(breakdown.compute_microseconds >= 0);
}

void should_name_every_node_class() {
  EXPECT_TRUE(mlx_plan::node_class_name(NodeClass::compute) == "compute");
  EXPECT_TRUE(
      mlx_plan::node_class_name(NodeClass::wait_on_cpu) == "wait_on_cpu");
  EXPECT_TRUE(
      mlx_plan::node_class_name(NodeClass::wait_on_gpu) == "wait_on_gpu");
  EXPECT_TRUE(
      mlx_plan::node_class_name(NodeClass::wait_on_dependency) ==
      "wait_on_dependency");
  EXPECT_TRUE(
      mlx_plan::node_class_name(NodeClass::scheduler_delay) ==
      "scheduler_delay");
}

// --- GPU-window mode: positive ---

void should_measure_compute_from_gpu_window_and_split_out_result_latency() {
  // issue 0→2, GPU runs 5→9, host notices at 12.
  const LatencyBreakdown breakdown = decompose_node_latency(
      with_gpu_window(make_timestamps(0, 2, 0, 0, 12), 5, 9));
  EXPECT_TRUE(breakdown.is_compute_measured);
  EXPECT_NEAR(breakdown.wait_on_cpu_microseconds, 2, EXACT_MICROSECONDS);
  EXPECT_NEAR(breakdown.wait_on_gpu_microseconds, 0, EXACT_MICROSECONDS);
  EXPECT_NEAR(breakdown.scheduler_delay_microseconds, 3, EXACT_MICROSECONDS);
  EXPECT_NEAR(breakdown.compute_microseconds, 4, EXACT_MICROSECONDS);
  EXPECT_NEAR(breakdown.completion_latency_microseconds, 3, EXACT_MICROSECONDS);
  EXPECT_NEAR(sum_of_slices(breakdown), 12, EXACT_MICROSECONDS);
}

void should_exclude_result_latency_from_classification() {
  // A tiny kernel noticed very late is still a compute node, not "waiting".
  const LatencyBreakdown breakdown = decompose_node_latency(
      with_gpu_window(make_timestamps(0, 1, 0, 0, 100), 1, 6));
  EXPECT_NEAR(breakdown.completion_latency_microseconds, 94, EXACT_MICROSECONDS);
  EXPECT_TRUE(breakdown.classification == NodeClass::compute);
}

void should_separate_stream_queueing_from_scheduler_delay() {
  // Stream busy until 6, GPU starts the buffer at 9: 5 µs queued + 3 µs
  // submission delay, after 1 µs of issue.
  const LatencyBreakdown breakdown = decompose_node_latency(
      with_gpu_window(make_timestamps(0, 1, 6, 0, 12), 9, 11));
  EXPECT_NEAR(breakdown.wait_on_gpu_microseconds, 5, EXACT_MICROSECONDS);
  EXPECT_NEAR(breakdown.scheduler_delay_microseconds, 3, EXACT_MICROSECONDS);
  EXPECT_TRUE(breakdown.classification == NodeClass::wait_on_gpu);
}

// --- GPU-window mode: edge / boundary ---

void should_charge_remote_wait_inside_gpu_window_to_dependency() {
  // The buffer starts at 2 but its head waits on a remote producer until 8.
  const LatencyBreakdown breakdown = decompose_node_latency(
      with_gpu_window(make_timestamps(0, 1, 0, 8, 12), 2, 10));
  EXPECT_NEAR(breakdown.wait_on_dependency_microseconds, 7, EXACT_MICROSECONDS);
  EXPECT_NEAR(breakdown.compute_microseconds, 2, EXACT_MICROSECONDS);
  EXPECT_TRUE(breakdown.classification == NodeClass::wait_on_dependency);
}

void should_credit_gpu_start_before_issue_returned_to_compute() {
  // Commit happens inside issue, so the GPU may start before issue returns.
  const LatencyBreakdown breakdown = decompose_node_latency(
      with_gpu_window(make_timestamps(0, 10, 0, 0, 20), 6, 18));
  EXPECT_NEAR(breakdown.wait_on_cpu_microseconds, 6, EXACT_MICROSECONDS);
  EXPECT_NEAR(breakdown.compute_microseconds, 12, EXACT_MICROSECONDS);
}

// --- GPU-window mode: negative ---

void should_clamp_gpu_window_outside_host_window() {
  // Clock-anchor error could place the GPU window outside the host window;
  // slices must stay non-negative and still sum to the wall.
  const LatencyBreakdown breakdown = decompose_node_latency(
      with_gpu_window(make_timestamps(10, 12, 0, 0, 20), 5, 30));
  EXPECT_TRUE(breakdown.wait_on_cpu_microseconds >= 0);
  EXPECT_TRUE(breakdown.compute_microseconds >= 0);
  EXPECT_TRUE(breakdown.completion_latency_microseconds >= 0);
  EXPECT_NEAR(sum_of_slices(breakdown), 10, EXACT_MICROSECONDS);
}

void should_report_host_residual_compute_as_unmeasured() {
  const LatencyBreakdown breakdown =
      decompose_node_latency(make_timestamps(0, 1, 0, 0, 11));
  EXPECT_TRUE(!breakdown.is_compute_measured);
  EXPECT_NEAR(breakdown.completion_latency_microseconds, 0, EXACT_MICROSECONDS);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_attribute_residual_to_compute_when_nothing_blocks),
      TEST_CASE(should_attribute_cross_stream_wait_to_dependency_at_a_join),
      TEST_CASE(should_attribute_stream_queueing_to_wait_on_gpu),
      TEST_CASE(should_classify_host_bound_node_as_wait_on_cpu),
      TEST_CASE(should_make_slices_sum_to_wall),
      TEST_CASE(should_classify_tie_between_waiting_and_compute_as_compute),
      TEST_CASE(should_not_charge_dependency_wait_when_remote_data_arrived_first),
      TEST_CASE(should_yield_all_zero_slices_for_zero_length_window),
      TEST_CASE(should_clamp_ready_instant_after_completion_without_negative_compute),
      TEST_CASE(should_report_zero_wall_when_completion_precedes_issue),
      TEST_CASE(should_name_every_node_class),
      TEST_CASE(should_measure_compute_from_gpu_window_and_split_out_result_latency),
      TEST_CASE(should_exclude_result_latency_from_classification),
      TEST_CASE(should_separate_stream_queueing_from_scheduler_delay),
      TEST_CASE(should_charge_remote_wait_inside_gpu_window_to_dependency),
      TEST_CASE(should_credit_gpu_start_before_issue_returned_to_compute),
      TEST_CASE(should_clamp_gpu_window_outside_host_window),
      TEST_CASE(should_report_host_residual_compute_as_unmeasured),
  });
}
