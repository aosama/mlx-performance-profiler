// Tests for timeline_dependencies: dependency snapshots on real (unevaluated)
// MLX graphs, and the pure readiness arithmetic both timeline passes share.

#include "graph_model.h"
#include "test_harness.h"
#include "timeline_dependencies.h"

#include "mlx/mlx.h"

#include <stdexcept>
#include <vector>

namespace {

using namespace mlx::core;
using mlx_plan::GpuExecutionWindow;

// --- positive ---

void should_split_inputs_by_stream_and_chain_each_stream() {
  const Stream main_stream = default_stream(Device::gpu);
  const Stream side_stream = new_stream(Device::gpu);
  const array side_value = exp(ones({8}, float32, side_stream), side_stream);
  const array main_value = exp(ones({8}, float32, main_stream), main_stream);
  const array joined = add(side_value, main_value, main_stream);
  const mlx_plan::CapturedGraph job_graph = mlx_plan::capture_graph({joined});
  const auto position_to_dependencies = mlx_plan::snapshot_dependencies(job_graph);
  const auto& join_dependencies = position_to_dependencies.back();
  EXPECT_TRUE(join_dependencies.cross_stream_input_positions.size() == 1);
  EXPECT_TRUE(join_dependencies.same_stream_input_positions.size() == 1);
  EXPECT_TRUE(join_dependencies.previous_on_stream_position != mlx_plan::NO_POSITION);
  const auto position_to_inputs = mlx_plan::merge_input_positions(position_to_dependencies);
  EXPECT_TRUE(position_to_inputs.back().size() == 2);
  eval(joined);
}

void should_prefer_measured_gpu_end_as_readiness() {
  GpuExecutionWindow measured_window;
  measured_window.is_measured = true;
  measured_window.finished_microseconds = 40;
  const std::vector<double> ready = mlx_plan::build_ready_microseconds(
      {50, 70}, {measured_window, GpuExecutionWindow{}});
  EXPECT_TRUE(ready == (std::vector<double>{40, 70}));
  EXPECT_TRUE(mlx_plan::latest_ready({0, 1}, ready) == 70);
}

// --- edge / boundary ---

void should_treat_no_producers_as_ready_at_pass_start() {
  EXPECT_TRUE(mlx_plan::latest_ready({}, {5, 6}) == 0);
  EXPECT_TRUE(mlx_plan::merge_input_positions({}).empty());
}

// --- negative ---

void should_reject_a_graph_with_an_unknown_stream() {
  const Stream gpu_stream = new_stream(Device::gpu);
  const array job_output = exp(ones({4}, float32, gpu_stream), gpu_stream);
  const mlx_plan::CapturedGraph job_graph = mlx_plan::capture_graph({job_output});
  bool has_thrown = false;
  try {
    mlx_plan::require_every_stream_known(job_graph, {});
  } catch (const std::out_of_range&) {
    has_thrown = true;
  }
  EXPECT_TRUE(has_thrown);
  eval(job_output);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_split_inputs_by_stream_and_chain_each_stream),
      TEST_CASE(should_prefer_measured_gpu_end_as_readiness),
      TEST_CASE(should_treat_no_producers_as_ready_at_pass_start),
      TEST_CASE(should_reject_a_graph_with_an_unknown_stream),
  });
}
