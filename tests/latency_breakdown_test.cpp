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

double sum_of_slices(const LatencyBreakdown& breakdown) {
  return breakdown.wait_on_cpu_microseconds +
      breakdown.wait_on_gpu_microseconds +
      breakdown.wait_on_dependency_microseconds +
      breakdown.compute_microseconds;
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
  });
}
