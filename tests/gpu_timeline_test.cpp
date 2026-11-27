// Integration tests for gpu_timeline: drives real MLX streams on the Metal
// GPU and checks the invariants the measurement must hold regardless of how
// fast this particular machine is. Timings themselves are not asserted —
// they vary run to run; structure and conservation laws do not.

#include "gpu_timeline.h"
#include "graph_model.h"
#include "backend_expectations.h"
#include "dispatch_recorder.h"
#include "telemetry.h"
#include "test_harness.h"

#include "mlx/mlx.h"

#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

using mlx_perf::install_dispatch_recorder;

using namespace mlx::core;
using mlx_perf::GpuTimeline;
using mlx_perf::NodeTimeline;

constexpr double SLICE_SUM_TOLERANCE_MICROSECONDS = 1e-6;

struct TwoStreamJob {
  std::vector<array> output_nodes;
  std::unordered_map<int, Stream> stream_index_to_stream;
};

// Two independent branches on two side streams, joined on the main stream:
// exactly one cross-stream join (the add), each branch fully stream-local.
TwoStreamJob build_two_stream_job() {
  const Stream main_stream = default_stream(default_device());
  const Stream left_stream = new_stream(default_device());
  const Stream right_stream = new_stream(default_device());
  const array left_branch =
      matmul(ones({128, 128}, float32, left_stream),
             ones({128, 128}, float32, left_stream), left_stream);
  const array right_branch = exp(zeros({128, 128}, float32, right_stream),
                                 right_stream);
  const array joined = add(left_branch, right_branch, main_stream);
  TwoStreamJob job{{joined}, {}};
  for (const Stream& job_stream : {main_stream, left_stream, right_stream}) {
    job.stream_index_to_stream.emplace(job_stream.index, job_stream);
  }
  return job;
}

// Which nodes launch a kernel, from the serial pass of an identical graph:
// on CUDA only those submit GPU work, so only those get a GPU window.
std::vector<bool> find_nodes_that_launch_kernels(const TwoStreamJob& job) {
  install_dispatch_recorder();
  const mlx_perf::CapturedGraph serial_graph = mlx_perf::capture_graph(job.output_nodes);
  const mlx_perf::RunTelemetry serial_telemetry =
      mlx_perf::run_attributed_pass(serial_graph, job.stream_index_to_stream);
  std::vector<bool> position_to_launches_kernels;
  for (const mlx_perf::NodeTelemetry& node_telemetry : serial_telemetry.attributed_nodes) {
    position_to_launches_kernels.push_back(!node_telemetry.kernel_dispatches.empty());
  }
  return position_to_launches_kernels;
}

void should_observe_every_node_complete_with_slices_summing_to_wall() {
  const std::vector<bool> position_to_launches_kernels = find_nodes_that_launch_kernels(build_two_stream_job());
  const TwoStreamJob job = build_two_stream_job();
  const mlx_perf::CapturedGraph job_graph = mlx_perf::capture_graph(job.output_nodes);
  const GpuTimeline gpu_timeline =
      mlx_perf::capture_gpu_timeline(job_graph, job.stream_index_to_stream);
  EXPECT_TRUE(gpu_timeline.nodes.size() == job_graph.topological.size());
  EXPECT_TRUE(position_to_launches_kernels.size() == gpu_timeline.nodes.size());
  int submitting_node_count = 0;
  for (std::size_t position = 0; position < gpu_timeline.nodes.size(); ++position) {
    const NodeTimeline& node_timeline = gpu_timeline.nodes[position];
    const auto& breakdown = node_timeline.breakdown;
    EXPECT_TRUE(node_timeline.node->handle.is_available());
    EXPECT_TRUE(
        node_timeline.timestamps.completed_microseconds >=
        node_timeline.timestamps.issue_finished_microseconds);
    EXPECT_NEAR(
        breakdown.wait_on_cpu_microseconds + breakdown.wait_on_gpu_microseconds +
            breakdown.wait_on_dependency_microseconds +
            breakdown.scheduler_delay_microseconds +
            breakdown.compute_microseconds +
            breakdown.completion_latency_microseconds,
        breakdown.wall_microseconds, SLICE_SUM_TOLERANCE_MICROSECONDS);
    // Every node here runs on a GPU stream. On Metal each carries a measured
    // GPU window; on CUDA each that launched a kernel does, and a node that
    // launched none has none.
    const bool submits_gpu_work = backend_expectations::DOES_EVERY_GPU_NODE_SUBMIT_WORK ||
        (position < position_to_launches_kernels.size() && position_to_launches_kernels[position]);
    submitting_node_count += submits_gpu_work ? 1 : 0;
    EXPECT_TRUE(node_timeline.timestamps.has_gpu_window == submits_gpu_work);
    EXPECT_TRUE(breakdown.is_compute_measured == submits_gpu_work);
    EXPECT_TRUE(
        node_timeline.timestamps.gpu_finished_microseconds >=
        node_timeline.timestamps.gpu_started_microseconds);
  }
  // One async_eval per GPU node that submits work: an exact count.
  EXPECT_TRUE(gpu_timeline.observed_command_buffer_count == submitting_node_count);
  EXPECT_TRUE(!gpu_timeline.is_command_buffer_count_lower_bound);
}

void should_flag_only_the_join_as_cross_stream_input() {
  const TwoStreamJob job = build_two_stream_job();
  const mlx_perf::CapturedGraph job_graph = mlx_perf::capture_graph(job.output_nodes);
  const GpuTimeline gpu_timeline =
      mlx_perf::capture_gpu_timeline(job_graph, job.stream_index_to_stream);
  int join_count = 0;
  for (const NodeTimeline& node_timeline : gpu_timeline.nodes) {
    if (node_timeline.has_cross_stream_input) {
      ++join_count;
      EXPECT_TRUE(node_timeline.node->op_name == "Add");
    } else {
      // A node with no cross-stream input can never be charged a
      // dependency wait: that slice is reserved for remote data.
      EXPECT_TRUE(node_timeline.breakdown.wait_on_dependency_microseconds == 0);
    }
  }
  EXPECT_TRUE(join_count == 1);
}

void should_report_one_stream_entry_per_stream_used_and_respect_join_order() {
  const TwoStreamJob job = build_two_stream_job();
  const mlx_perf::CapturedGraph job_graph = mlx_perf::capture_graph(job.output_nodes);
  const GpuTimeline gpu_timeline =
      mlx_perf::capture_gpu_timeline(job_graph, job.stream_index_to_stream);
  EXPECT_TRUE(gpu_timeline.streams.size() == 3);
  const NodeTimeline& join_timeline = gpu_timeline.nodes.back();
  for (const NodeTimeline& node_timeline : gpu_timeline.nodes) {
    // The join is last in topological order and consumes everything, so no
    // producer may be observed completing after it.
    EXPECT_TRUE(
        node_timeline.timestamps.completed_microseconds <=
        join_timeline.timestamps.completed_microseconds);
  }
  EXPECT_TRUE(gpu_timeline.timeline_wall_microseconds > 0);
}

// Edge: divmod is ONE primitive with two sibling outputs. Consuming both
// makes the capture see the second output only as a sibling, so the join's
// remote readiness must still resolve to the DivMod node's completion.
void should_resolve_sibling_outputs_to_their_shared_primitive() {
  const Stream main_stream = default_stream(default_device());
  const Stream side_stream = new_stream(default_device());
  const std::vector<array> quotient_and_remainder = divmod(
      full({64}, 7.0f, side_stream), full({64}, 3.0f, side_stream), side_stream);
  const array joined =
      add(quotient_and_remainder[0], quotient_and_remainder[1], main_stream);
  const std::unordered_map<int, Stream> stream_index_to_stream = {
      {main_stream.index, main_stream}, {side_stream.index, side_stream}};
  const mlx_perf::CapturedGraph job_graph = mlx_perf::capture_graph({joined});
  const GpuTimeline gpu_timeline =
      mlx_perf::capture_gpu_timeline(job_graph, stream_index_to_stream);
  const NodeTimeline* divmod_timeline = nullptr;
  for (const NodeTimeline& node_timeline : gpu_timeline.nodes) {
    if (node_timeline.node->op_name == "DivMod") {
      divmod_timeline = &node_timeline;
    }
  }
  EXPECT_TRUE(divmod_timeline != nullptr);
  const NodeTimeline& join_timeline = gpu_timeline.nodes.back();
  EXPECT_TRUE(join_timeline.has_cross_stream_input);
  if (divmod_timeline != nullptr) {
    // Readiness is the producer's GPU end (exact), not when the host
    // happened to notice it.
    EXPECT_TRUE(divmod_timeline->timestamps.has_gpu_window);
    EXPECT_TRUE(
        join_timeline.timestamps.cross_stream_inputs_ready_microseconds ==
        divmod_timeline->timestamps.gpu_finished_microseconds);
  }
}

// Edge: a CPU-device stream must be reported as such, not assumed GPU.
void should_report_cpu_stream_as_cpu_device() {
  const Stream cpu_stream = new_stream(Device::cpu);
  const array cpu_output =
      exp(ones({16}, float32, cpu_stream), cpu_stream);
  const std::unordered_map<int, Stream> stream_index_to_stream = {
      {cpu_stream.index, cpu_stream}};
  const mlx_perf::CapturedGraph job_graph = mlx_perf::capture_graph({cpu_output});
  const GpuTimeline gpu_timeline =
      mlx_perf::capture_gpu_timeline(job_graph, stream_index_to_stream);
  EXPECT_TRUE(gpu_timeline.streams.size() == 1);
  if (!gpu_timeline.streams.empty()) {
    EXPECT_TRUE(!gpu_timeline.streams.front().is_gpu);
  }
  for (const NodeTimeline& node_timeline : gpu_timeline.nodes) {
    EXPECT_TRUE(!node_timeline.timestamps.has_gpu_window);
  }
  // CPU streams have no command buffers to count.
  EXPECT_TRUE(gpu_timeline.observed_command_buffer_count == 0);
  EXPECT_TRUE(cpu_output.is_available());
}

// Boundary: a graph whose output is already materialized has no primitives
// to dispatch; the pass must return an empty timeline, not spin forever.
void should_return_empty_timeline_for_already_materialized_graph() {
  array materialized_output = ones({4}, float32);
  materialized_output.eval();
  const mlx_perf::CapturedGraph job_graph =
      mlx_perf::capture_graph({materialized_output});
  const std::unordered_map<int, Stream> stream_index_to_stream;
  const GpuTimeline gpu_timeline =
      mlx_perf::capture_gpu_timeline(job_graph, stream_index_to_stream);
  EXPECT_TRUE(gpu_timeline.nodes.empty());
  EXPECT_TRUE(gpu_timeline.streams.empty());
  EXPECT_TRUE(gpu_timeline.timeline_wall_microseconds == 0);
  EXPECT_TRUE(gpu_timeline.observed_command_buffer_count == 0);
}

// Negative: a stream the caller did not supply is a caller error that must
// surface loudly rather than be silently attributed to some other device.
void should_throw_when_a_node_stream_is_missing_from_the_stream_map() {
  const TwoStreamJob job = build_two_stream_job();
  const mlx_perf::CapturedGraph job_graph = mlx_perf::capture_graph(job.output_nodes);
  const std::unordered_map<int, Stream> incomplete_stream_map;
  bool has_thrown = false;
  try {
    mlx_perf::capture_gpu_timeline(job_graph, incomplete_stream_map);
  } catch (const std::out_of_range&) {
    has_thrown = true;
  }
  EXPECT_TRUE(has_thrown);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_observe_every_node_complete_with_slices_summing_to_wall),
      TEST_CASE(should_flag_only_the_join_as_cross_stream_input),
      TEST_CASE(should_report_one_stream_entry_per_stream_used_and_respect_join_order),
      TEST_CASE(should_resolve_sibling_outputs_to_their_shared_primitive),
      TEST_CASE(should_report_cpu_stream_as_cpu_device),
      TEST_CASE(should_return_empty_timeline_for_already_materialized_graph),
      TEST_CASE(should_throw_when_a_node_stream_is_missing_from_the_stream_map),
  });
}
