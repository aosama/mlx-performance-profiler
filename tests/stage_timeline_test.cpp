// Integration tests for stage_timeline: real MLX streams on the Metal GPU,
// timed one async_eval per stage. Like gpu_timeline_test, timings are never
// asserted exactly — only structure, conservation laws and bounds that hold
// on any machine.

#include "graph_factory.h"
#include "graph_model.h"
#include "stage_timeline.h"
#include "test_harness.h"

#include "mlx/mlx.h"

#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

using namespace mlx::core;
using mlx_perf::GpuTimeline;
using mlx_perf::StageTimeline;

constexpr double SLICE_SUM_TOLERANCE_MICROSECONDS = 1e-6;
// Far more ops than MLX's per-buffer budget (20–50), so the stage MUST span
// several command buffers and only the end marker can see its true end.
constexpr int MULTI_BUFFER_CHAIN_LENGTH = 300;

struct StagedJob {
  std::vector<array> output_arrays;
  std::vector<mlx_perf::StageBoundary> stage_boundaries;
  std::unordered_map<int, Stream> stream_index_to_stream;
};

// Two "layers" of matmul+exp on one GPU stream, then a head reduction.
StagedJob build_two_layer_job() {
  const Stream gpu_stream = new_stream(Device::gpu);
  const array job_input = ones({128, 128}, float32, gpu_stream);
  const array layer_zero = exp(matmul(job_input, job_input, gpu_stream), gpu_stream);
  const array layer_one = exp(matmul(layer_zero, job_input, gpu_stream), gpu_stream);
  const array head = sum(layer_one, false, gpu_stream);
  return {{head}, {{"layer 0", {layer_zero}}, {"layer 1", {layer_one}}},
          {{gpu_stream.index, gpu_stream}}};
}

GpuTimeline capture_staged(const StagedJob& job, const mlx_perf::CapturedGraph& job_graph) {
  return mlx_perf::capture_stage_timeline(
      job_graph, job.stream_index_to_stream,
      mlx_perf::resolve_stage_cuts(job_graph, job.stage_boundaries), "head");
}

void expect_slices_sum_to_wall(const StageTimeline& stage_task) {
  const auto& breakdown = stage_task.breakdown;
  EXPECT_NEAR(breakdown.wait_on_cpu_microseconds + breakdown.wait_on_gpu_microseconds +
                  breakdown.wait_on_dependency_microseconds +
                  breakdown.scheduler_delay_microseconds + breakdown.compute_microseconds +
                  breakdown.completion_latency_microseconds,
              breakdown.wall_microseconds, SLICE_SUM_TOLERANCE_MICROSECONDS);
}

// --- positive ---

void should_time_one_task_per_stage_with_slices_summing_to_wall() {
  const StagedJob job = build_two_layer_job();
  const mlx_perf::CapturedGraph job_graph = mlx_perf::capture_graph(job.output_arrays);
  const GpuTimeline gpu_timeline = capture_staged(job, job_graph);
  EXPECT_TRUE(gpu_timeline.granularity == mlx_perf::TimelineGranularity::per_stage);
  EXPECT_TRUE(gpu_timeline.stages.size() == 3);
  if (gpu_timeline.stages.size() == 3) {
    EXPECT_TRUE(gpu_timeline.stages[0].stage_name == "layer 0");
    EXPECT_TRUE(gpu_timeline.stages[2].stage_name == "head");
  }
  for (const StageTimeline& stage_task : gpu_timeline.stages) {
    expect_slices_sum_to_wall(stage_task);
    EXPECT_TRUE(stage_task.breakdown.is_compute_measured);
    // The GPU window sits inside the host window: it cannot end after the
    // host saw the stage complete (the end bound guarantees it).
    EXPECT_TRUE(stage_task.timestamps.gpu_finished_microseconds <=
                stage_task.timestamps.completed_microseconds);
  }
  EXPECT_TRUE(gpu_timeline.timeline_wall_microseconds > 0);
  EXPECT_TRUE(gpu_timeline.gpu_measured_node_count ==
              static_cast<int>(job_graph.topological.size()));
}

// Every node is evaluated, belongs to exactly one stage task, and carries no
// per-node timing (the pass never measured one).
void should_partition_every_node_into_exactly_one_stage_task() {
  const StagedJob job = build_two_layer_job();
  const mlx_perf::CapturedGraph job_graph = mlx_perf::capture_graph(job.output_arrays);
  const GpuTimeline gpu_timeline = capture_staged(job, job_graph);
  std::vector<int> position_to_membership_count(gpu_timeline.nodes.size(), 0);
  for (std::size_t task_position = 0; task_position < gpu_timeline.stages.size();
       ++task_position) {
    for (const int member_position : gpu_timeline.stages[task_position].node_positions) {
      ++position_to_membership_count[member_position];
      EXPECT_TRUE(gpu_timeline.nodes[member_position].stage_position ==
                  static_cast<int>(task_position));
    }
  }
  for (std::size_t position = 0; position < gpu_timeline.nodes.size(); ++position) {
    EXPECT_TRUE(position_to_membership_count[position] == 1);
    EXPECT_TRUE(gpu_timeline.nodes[position].node->handle.is_available());
    EXPECT_TRUE(gpu_timeline.nodes[position].breakdown.wall_microseconds == 0);
  }
}

// One stage of 300 dependent adds spans several command buffers. Without
// the end marker the window would cover only the first buffer (~1/6 of the
// GPU work); with it, compute must cover most of the time the GPU was busy
// after the host finished issuing.
void should_cover_every_command_buffer_of_a_multi_buffer_stage() {
  const Stream gpu_stream = new_stream(Device::gpu);
  array chain = random::normal({1 << 20}, float32, 0.0f, 1.0f, std::nullopt, gpu_stream);
  eval(chain);
  for (int add_index = 0; add_index < MULTI_BUFFER_CHAIN_LENGTH; ++add_index) {
    chain = add(chain, array(1.0f), gpu_stream);
  }
  const mlx_perf::CapturedGraph job_graph = mlx_perf::capture_graph({chain});
  const GpuTimeline gpu_timeline = mlx_perf::capture_stage_timeline(
      job_graph, {{gpu_stream.index, gpu_stream}}, {}, "whole job");
  EXPECT_TRUE(gpu_timeline.stages.size() == 1);
  if (gpu_timeline.stages.size() == 1) {
    const StageTimeline& stage_task = gpu_timeline.stages.front();
    const double after_issue_microseconds = stage_task.timestamps.completed_microseconds -
        stage_task.timestamps.issue_finished_microseconds;
    EXPECT_TRUE(stage_task.breakdown.is_compute_measured);
    EXPECT_TRUE(stage_task.breakdown.compute_microseconds >= 0.5 * after_issue_microseconds);
    expect_slices_sum_to_wall(stage_task);
  }
}

// A join on the main stream in a later stage reads a side-stream stage: the
// task must name that producer, and only that task is a join.
void should_link_cross_stream_stage_tasks_across_stages() {
  const Stream main_stream = default_stream(Device::gpu);
  const Stream side_stream = new_stream(Device::gpu);
  const array side_branch = exp(ones({64, 64}, float32, side_stream), side_stream);
  const array main_branch = exp(zeros({64, 64}, float32, main_stream), main_stream);
  const array joined = add(side_branch, main_branch, main_stream);
  const mlx_perf::CapturedGraph job_graph = mlx_perf::capture_graph({joined});
  const GpuTimeline gpu_timeline = mlx_perf::capture_stage_timeline(
      job_graph, {{main_stream.index, main_stream}, {side_stream.index, side_stream}},
      mlx_perf::resolve_stage_cuts(job_graph, {{"branches", {side_branch, main_branch}}}),
      "join");
  // Stage "branches" runs on two streams (two tasks); "join" on one.
  EXPECT_TRUE(gpu_timeline.stages.size() == 3);
  int join_task_count = 0;
  for (const StageTimeline& stage_task : gpu_timeline.stages) {
    if (stage_task.has_cross_stream_input) {
      ++join_task_count;
      EXPECT_TRUE(stage_task.stage_name == "join");
      EXPECT_TRUE(stage_task.cross_stream_producer_stage_positions.size() == 1);
    } else {
      EXPECT_TRUE(stage_task.breakdown.wait_on_dependency_microseconds == 0);
    }
  }
  EXPECT_TRUE(join_task_count == 1);
  EXPECT_TRUE(gpu_timeline.streams.size() == 2);
}

// --- edge / boundary ---

void should_skip_boundaries_that_are_not_in_the_graph() {
  const StagedJob job = build_two_layer_job();
  const mlx_perf::CapturedGraph job_graph = mlx_perf::capture_graph(job.output_arrays);
  array unrelated = ones({2});
  const std::vector<mlx_perf::StageCut> stage_cuts =
      mlx_perf::resolve_stage_cuts(job_graph, {{"unrelated", {unrelated}}});
  EXPECT_TRUE(stage_cuts.size() == 1);
  EXPECT_TRUE(stage_cuts.front().boundary_positions.empty());
  eval(job.output_arrays);
}

void should_leave_cpu_stream_stage_unmeasured_but_complete() {
  const Stream cpu_stream = new_stream(Device::cpu);
  const array cpu_output = exp(ones({16}, float32, cpu_stream), cpu_stream);
  const mlx_perf::CapturedGraph job_graph = mlx_perf::capture_graph({cpu_output});
  const GpuTimeline gpu_timeline = mlx_perf::capture_stage_timeline(
      job_graph, {{cpu_stream.index, cpu_stream}}, {}, "whole job");
  EXPECT_TRUE(gpu_timeline.stages.size() == 1);
  if (!gpu_timeline.stages.empty()) {
    EXPECT_TRUE(!gpu_timeline.stages.front().breakdown.is_compute_measured);
    expect_slices_sum_to_wall(gpu_timeline.stages.front());
  }
  EXPECT_TRUE(cpu_output.is_available());
  EXPECT_TRUE(gpu_timeline.streams.size() == 1 && !gpu_timeline.streams.front().is_gpu);
}

void should_return_empty_timeline_for_already_materialized_graph() {
  array materialized_output = ones({4}, float32);
  materialized_output.eval();
  const mlx_perf::CapturedGraph job_graph = mlx_perf::capture_graph({materialized_output});
  const GpuTimeline gpu_timeline =
      mlx_perf::capture_stage_timeline(job_graph, {}, {}, "whole job");
  EXPECT_TRUE(gpu_timeline.stages.empty());
  EXPECT_TRUE(gpu_timeline.nodes.empty());
  EXPECT_TRUE(gpu_timeline.timeline_wall_microseconds == 0);
}

// --- negative ---

void should_throw_before_dispatch_when_a_stream_is_missing() {
  const StagedJob job = build_two_layer_job();
  const mlx_perf::CapturedGraph job_graph = mlx_perf::capture_graph(job.output_arrays);
  bool has_thrown = false;
  try {
    mlx_perf::capture_stage_timeline(job_graph, {}, {}, "whole job");
  } catch (const std::out_of_range&) {
    has_thrown = true;
  }
  EXPECT_TRUE(has_thrown);
  // Failing fast must leave the job unevaluated for the caller.
  EXPECT_TRUE(!job.output_arrays.front().is_available());
}

} // namespace

int main() {
  return test_harness::run_all({
      {"should_time_one_task_per_stage_with_slices_summing_to_wall",
       should_time_one_task_per_stage_with_slices_summing_to_wall},
      {"should_partition_every_node_into_exactly_one_stage_task",
       should_partition_every_node_into_exactly_one_stage_task},
      {"should_cover_every_command_buffer_of_a_multi_buffer_stage",
       should_cover_every_command_buffer_of_a_multi_buffer_stage},
      {"should_link_cross_stream_stage_tasks_across_stages",
       should_link_cross_stream_stage_tasks_across_stages},
      {"should_skip_boundaries_that_are_not_in_the_graph",
       should_skip_boundaries_that_are_not_in_the_graph},
      {"should_leave_cpu_stream_stage_unmeasured_but_complete",
       should_leave_cpu_stream_stage_unmeasured_but_complete},
      {"should_return_empty_timeline_for_already_materialized_graph",
       should_return_empty_timeline_for_already_materialized_graph},
      {"should_throw_before_dispatch_when_a_stream_is_missing",
       should_throw_before_dispatch_when_a_stream_is_missing},
  });
}
