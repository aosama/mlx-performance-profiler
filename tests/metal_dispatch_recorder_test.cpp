// Integration tests for metal_dispatch_recorder: real MLX launches on the
// Metal GPU, recorded with their exact shapes and names.

#include "metal_dispatch_recorder.h"
#include "test_harness.h"

#include "mlx/fast.h"
#include "mlx/mlx.h"

#include <string>
#include <thread>

namespace {

using namespace mlx::core;
using mlx_perf::DispatchGridUnit;
using mlx_perf::DispatchRecordingScope;
using mlx_perf::KernelDispatch;

// Doubles `thread_count` floats with a custom kernel launched as
// `thread_count` threads in threadgroups of `threads_per_threadgroup`.
array double_with_custom_kernel(const std::string& author_kernel_name, const array& input_values,
                                int thread_count, int threads_per_threadgroup) {
  const auto custom_kernel = fast::metal_kernel(
      author_kernel_name, {"input_values"}, {"output_values"},
      "uint element_index = thread_position_in_grid.x;\n"
      "output_values[element_index] = 2.0f * input_values[element_index];\n");
  return custom_kernel({input_values}, {input_values.shape()}, {float32}, {thread_count, 1, 1},
                       {threads_per_threadgroup, 1, 1}, {}, std::nullopt, false, {})[0];
}

std::vector<KernelDispatch> record_evaluation(array output_array) {
  DispatchRecordingScope recording_scope;
  eval(output_array);
  synchronize();
  return recording_scope.take_recorded_dispatches();
}

// --- positive ---

void should_install_the_hooks_on_this_gpu() {
  EXPECT_TRUE(mlx_perf::install_dispatch_recorder());
  EXPECT_TRUE(mlx_perf::install_dispatch_recorder()); // idempotent
}

void should_record_a_custom_kernel_launch_with_its_exact_shape() {
  mlx_perf::install_dispatch_recorder();
  const array input_values = ones({128});
  eval(input_values);
  const auto recorded_dispatches =
      record_evaluation(double_with_custom_kernel("four_groups", input_values, 128, 32));
  EXPECT_TRUE(recorded_dispatches.size() == 1);
  const KernelDispatch& launch = recorded_dispatches.front();
  EXPECT_TRUE(launch.grid_unit == DispatchGridUnit::threads);
  EXPECT_TRUE(launch.grid.width == 128 && launch.threads_per_threadgroup.width == 32);
  EXPECT_TRUE(mlx_perf::count_threadgroups(launch) == 4);
  EXPECT_TRUE(launch.metal_function_name.find("four_groups") != std::string::npos);
  EXPECT_TRUE(launch.max_threads_per_threadgroup >= 32);
  EXPECT_TRUE(launch.thread_execution_width > 0);
}

void should_record_and_name_a_stock_mlx_kernel() {
  mlx_perf::install_dispatch_recorder();
  const array left = ones({256, 256});
  eval(left);
  const auto recorded_dispatches = record_evaluation(matmul(left, left));
  EXPECT_TRUE(!recorded_dispatches.empty());
  EXPECT_TRUE(mlx_perf::count_threadgroups(recorded_dispatches.front()) > 0);
  EXPECT_TRUE(!recorded_dispatches.front().metal_function_name.empty());
}

// --- edge / boundary ---

void should_record_into_the_innermost_scope_then_resume_the_outer() {
  mlx_perf::install_dispatch_recorder();
  const array input_values = ones({64});
  eval(input_values);
  DispatchRecordingScope outer_scope;
  {
    DispatchRecordingScope inner_scope;
    eval(double_with_custom_kernel("inner_launch", input_values, 64, 32));
    synchronize();
    EXPECT_TRUE(inner_scope.take_recorded_dispatches().size() == 1);
  }
  eval(double_with_custom_kernel("outer_launch", input_values, 64, 32));
  synchronize();
  const auto outer_dispatches = outer_scope.take_recorded_dispatches();
  EXPECT_TRUE(outer_dispatches.size() == 1);
  EXPECT_TRUE(outer_dispatches.front().metal_function_name.find("outer_launch") != std::string::npos);
}

// --- negative ---

void should_record_nothing_outside_a_scope_or_on_another_thread() {
  mlx_perf::install_dispatch_recorder();
  const array input_values = ones({64});
  eval(input_values);
  eval(double_with_custom_kernel("before_scope", input_values, 64, 32));
  synchronize();
  DispatchRecordingScope recording_scope;
  std::thread other_thread([&input_values]() {
    eval(double_with_custom_kernel("other_thread", input_values, 64, 32));
    synchronize();
  });
  other_thread.join();
  EXPECT_TRUE(recording_scope.take_recorded_dispatches().empty());
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_install_the_hooks_on_this_gpu),
      TEST_CASE(should_record_a_custom_kernel_launch_with_its_exact_shape),
      TEST_CASE(should_record_and_name_a_stock_mlx_kernel),
      TEST_CASE(should_record_into_the_innermost_scope_then_resume_the_outer),
      TEST_CASE(should_record_nothing_outside_a_scope_or_on_another_thread),
  });
}
