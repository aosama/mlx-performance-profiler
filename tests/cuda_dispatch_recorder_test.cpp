// Integration tests for cuda_dispatch_recorder: real MLX launches on the
// CUDA GPU, recorded with their exact shapes and names. The same cases as
// the Metal recorder's tests, so both backends meet one contract.

#include "dispatch_recorder.h"
#include "custom_kernel_fixture.h"
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
// `thread_count` threads in blocks of `threads_per_threadgroup` (MLX's CUDA
// custom kernels take the grid in threads, as Metal's dispatchThreads does,
// and launch it in blocks).
array double_with_custom_kernel(const std::string& author_kernel_name, const array& input_values,
                                int thread_count, int threads_per_threadgroup) {
  const auto custom_kernel = custom_kernel_fixture::make_doubling_kernel(author_kernel_name);
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

void should_install_the_observers_on_this_gpu() {
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
  // The driver receives the grid in blocks.
  EXPECT_TRUE(launch.grid_unit == DispatchGridUnit::threadgroups);
  EXPECT_TRUE(launch.grid.width == 4 && launch.threads_per_threadgroup.width == 32);
  EXPECT_TRUE(mlx_perf::count_threadgroups(launch) == 4);
  EXPECT_TRUE(launch.kernel_function_name.find("four_groups") != std::string::npos);
  EXPECT_TRUE(launch.max_threads_per_threadgroup >= 32);
  EXPECT_TRUE(launch.thread_execution_width == 32);
}

void should_record_and_name_a_stock_mlx_kernel() {
  mlx_perf::install_dispatch_recorder();
  const array left = ones({256, 256});
  eval(left);
  const auto recorded_dispatches = record_evaluation(matmul(left, left));
  EXPECT_TRUE(!recorded_dispatches.empty());
  EXPECT_TRUE(mlx_perf::count_threadgroups(recorded_dispatches.front()) > 0);
  EXPECT_TRUE(!recorded_dispatches.front().kernel_function_name.empty());
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
  EXPECT_TRUE(outer_dispatches.front().kernel_function_name.find("outer_launch") != std::string::npos);
}

// #44: a procedure that evaluates on a worker thread, recorded process-wide.
void should_record_another_threads_launches_in_a_whole_process_scope() {
  mlx_perf::install_dispatch_recorder();
  const array input_values = ones({64});
  eval(input_values);
  DispatchRecordingScope recording_scope(mlx_perf::RecordingReach::whole_process);
  std::thread worker_thread([&input_values]() {
    eval(double_with_custom_kernel("worker_thread", input_values, 64, 32));
    synchronize();
  });
  worker_thread.join();
  const std::vector<KernelDispatch> recorded_dispatches = recording_scope.take_recorded_dispatches();
  EXPECT_TRUE(recorded_dispatches.size() == 1);
  EXPECT_TRUE(!recorded_dispatches.empty() &&
              recorded_dispatches[0].kernel_function_name.find("worker_thread") != std::string::npos);
}

// --- edge / boundary ---

// The serial pass's per-thread scope keeps its node's launches to itself
// even while a whole-process scope is open.
void should_let_a_threads_own_scope_win_over_a_whole_process_one() {
  mlx_perf::install_dispatch_recorder();
  const array input_values = ones({64});
  eval(input_values);
  DispatchRecordingScope process_scope(mlx_perf::RecordingReach::whole_process);
  const std::vector<KernelDispatch> thread_dispatches =
      record_evaluation(double_with_custom_kernel("thread_scope_wins", input_values, 64, 32));
  EXPECT_TRUE(thread_dispatches.size() == 1);
  EXPECT_TRUE(process_scope.take_recorded_dispatches().empty());
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
      TEST_CASE(should_install_the_observers_on_this_gpu),
      TEST_CASE(should_record_a_custom_kernel_launch_with_its_exact_shape),
      TEST_CASE(should_record_and_name_a_stock_mlx_kernel),
      TEST_CASE(should_record_into_the_innermost_scope_then_resume_the_outer),
      TEST_CASE(should_record_another_threads_launches_in_a_whole_process_scope),
      TEST_CASE(should_let_a_threads_own_scope_win_over_a_whole_process_one),
      TEST_CASE(should_record_nothing_outside_a_scope_or_on_another_thread),
  });
}
