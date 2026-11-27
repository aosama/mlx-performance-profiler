// Unit tests for cuda_host_hand_off: which driver calls are submissions and
// which returns are hand-offs, from the call's id and result alone.

#include "cuda_host_hand_off.h"
#include "test_harness.h"

namespace {

using mlx_perf::counts_as_host_hand_off;
using mlx_perf::is_graph_launch;
using mlx_perf::is_host_wait_or_completion_check;

// --- positive ---

void should_count_every_wait_as_a_hand_off() {
  EXPECT_TRUE(counts_as_host_hand_off(CUPTI_DRIVER_TRACE_CBID_cuEventSynchronize, CUDA_SUCCESS));
  EXPECT_TRUE(counts_as_host_hand_off(CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize, CUDA_SUCCESS));
  EXPECT_TRUE(counts_as_host_hand_off(CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize_ptsz, CUDA_SUCCESS));
  EXPECT_TRUE(counts_as_host_hand_off(CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize_v2, CUDA_SUCCESS));
}

void should_count_a_completion_check_that_found_the_work_done() {
  EXPECT_TRUE(counts_as_host_hand_off(CUPTI_DRIVER_TRACE_CBID_cuEventQuery, CUDA_SUCCESS));
}

void should_recognise_both_graph_launch_forms() {
  EXPECT_TRUE(is_graph_launch(CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch));
  EXPECT_TRUE(is_graph_launch(CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz));
}

// --- edge / boundary ---

// A wait that returned an error still blocked the host until it returned.
void should_count_a_wait_that_returned_an_error() {
  EXPECT_TRUE(counts_as_host_hand_off(CUPTI_DRIVER_TRACE_CBID_cuEventSynchronize, CUDA_ERROR_LAUNCH_FAILED));
}

void should_keep_launches_and_waits_apart() {
  EXPECT_TRUE(!is_host_wait_or_completion_check(CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch));
  EXPECT_TRUE(!is_graph_launch(CUPTI_DRIVER_TRACE_CBID_cuEventQuery));
}

// --- negative ---

// cuEventQuery reports work still running as CUDA_ERROR_NOT_READY: a poll,
// not a hand-off.
void should_not_count_a_completion_check_that_found_the_work_running() {
  EXPECT_TRUE(!counts_as_host_hand_off(CUPTI_DRIVER_TRACE_CBID_cuEventQuery, CUDA_ERROR_NOT_READY));
}

void should_not_count_a_call_that_neither_waits_nor_checks() {
  EXPECT_TRUE(!counts_as_host_hand_off(CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch, CUDA_SUCCESS));
  EXPECT_TRUE(!counts_as_host_hand_off(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, CUDA_SUCCESS));
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_count_every_wait_as_a_hand_off),
      TEST_CASE(should_count_a_completion_check_that_found_the_work_done),
      TEST_CASE(should_recognise_both_graph_launch_forms),
      TEST_CASE(should_count_a_wait_that_returned_an_error),
      TEST_CASE(should_keep_launches_and_waits_apart),
      TEST_CASE(should_not_count_a_completion_check_that_found_the_work_running),
      TEST_CASE(should_not_count_a_call_that_neither_waits_nor_checks),
  });
}
