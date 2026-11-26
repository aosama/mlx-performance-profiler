// Integration tests for metal_submission_recorder: real MLX evaluations on
// the Metal GPU, their command buffer commits and blocking waits recorded.

#include "metal_host_clock.h"
#include "metal_submission_recorder.h"
#include "test_harness.h"

#include "mlx/mlx.h"

#include <thread>

namespace {

using namespace mlx::core;
using mlx_perf::SubmissionRecord;
using mlx_perf::SubmissionRecordingScope;

// `step_count` blocking evaluations of a growing chain: the shape of a
// procedure that evaluates inside itself.
void run_blocking_steps(int step_count) {
  array running_value = ones({256, 256});
  for (int step_index = 0; step_index < step_count; ++step_index) {
    running_value = running_value * 1.0001f + 1.0f;
    eval(running_value);
  }
}

// --- positive ---

void should_install_the_hooks_on_this_gpu() {
  EXPECT_TRUE(mlx_perf::install_submission_recorder());
  EXPECT_TRUE(mlx_perf::install_submission_recorder());
}

void should_count_each_blocking_evaluation_and_its_command_buffer() {
  mlx_perf::install_submission_recorder();
  run_blocking_steps(1); // warm the kernels outside the scope
  SubmissionRecordingScope recording_scope;
  const double started_seconds = mlx_perf::current_metal_host_seconds();
  run_blocking_steps(4);
  const SubmissionRecord submission_record = recording_scope.finish_and_take_record();
  const double finished_seconds = mlx_perf::current_metal_host_seconds();
  EXPECT_TRUE(submission_record.blocking_wait_count >= 4);
  EXPECT_TRUE(submission_record.committed_command_buffer_count >= 4);
  EXPECT_TRUE(static_cast<int>(submission_record.command_buffer_windows.size()) ==
              submission_record.committed_command_buffer_count);
  EXPECT_TRUE(submission_record.blocking_wait_microseconds > 0.0);
  EXPECT_TRUE(mlx_perf::measure_gpu_busy_microseconds(submission_record.command_buffer_windows,
                                                     started_seconds, finished_seconds) > 0.0);
}

void should_wait_for_unwaited_async_work_without_counting_that_wait() {
  mlx_perf::install_submission_recorder();
  SubmissionRecordingScope recording_scope;
  array unwaited_output = matmul(ones({512, 512}), ones({512, 512}));
  async_eval(unwaited_output);
  const SubmissionRecord submission_record = recording_scope.finish_and_take_record();
  EXPECT_TRUE(submission_record.committed_command_buffer_count >= 1);
  EXPECT_TRUE(submission_record.blocking_wait_count == 0);
  EXPECT_TRUE(static_cast<int>(submission_record.command_buffer_windows.size()) ==
              submission_record.committed_command_buffer_count);
}

// --- edge / boundary ---

void should_record_nothing_for_a_scope_without_mlx_work() {
  SubmissionRecordingScope recording_scope;
  const SubmissionRecord submission_record = recording_scope.finish_and_take_record();
  EXPECT_TRUE(submission_record.committed_command_buffer_count == 0);
  EXPECT_TRUE(submission_record.blocking_wait_count == 0);
  EXPECT_TRUE(submission_record.command_buffer_windows.empty());
}

void should_start_a_fresh_record_after_each_take() {
  mlx_perf::install_submission_recorder();
  SubmissionRecordingScope recording_scope;
  run_blocking_steps(2);
  recording_scope.finish_and_take_record();
  EXPECT_TRUE(recording_scope.finish_and_take_record().committed_command_buffer_count == 0);
}

// --- negative ---

void should_record_nothing_from_another_thread() {
  mlx_perf::install_submission_recorder();
  SubmissionRecordingScope recording_scope;
  std::thread other_thread([]() { run_blocking_steps(3); });
  other_thread.join();
  const SubmissionRecord submission_record = recording_scope.finish_and_take_record();
  EXPECT_TRUE(submission_record.committed_command_buffer_count == 0);
  EXPECT_TRUE(submission_record.blocking_wait_count == 0);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_install_the_hooks_on_this_gpu),
      TEST_CASE(should_count_each_blocking_evaluation_and_its_command_buffer),
      TEST_CASE(should_wait_for_unwaited_async_work_without_counting_that_wait),
      TEST_CASE(should_record_nothing_for_a_scope_without_mlx_work),
      TEST_CASE(should_start_a_fresh_record_after_each_take),
      TEST_CASE(should_record_nothing_from_another_thread),
  });
}
