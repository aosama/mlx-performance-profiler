// Unit tests for dispatch_occupancy: the threadgroups a launch forms and the
// share of GPU cores it can give work to. Pure arithmetic, no GPU.

#include "dispatch_occupancy.h"
#include "test_harness.h"

namespace {

using namespace mlx_perf;

KernelDispatch make_dispatch(DispatchGridUnit grid_unit, GridSize grid, GridSize threads_per_threadgroup) {
  KernelDispatch kernel_dispatch;
  kernel_dispatch.grid_unit = grid_unit;
  kernel_dispatch.grid = grid;
  kernel_dispatch.threads_per_threadgroup = threads_per_threadgroup;
  return kernel_dispatch;
}

// --- positive ---

void should_count_the_threadgroups_a_threadgroup_grid_names() {
  const KernelDispatch kernel_dispatch =
      make_dispatch(DispatchGridUnit::threadgroups, {8, 4, 2}, {32, 1, 1});
  EXPECT_TRUE(count_threadgroups(kernel_dispatch) == 64);
}

void should_form_threadgroups_from_a_thread_grid() {
  // 128 threads in groups of 32: the 4-threadgroup launch of issue #35.
  const KernelDispatch kernel_dispatch = make_dispatch(DispatchGridUnit::threads, {128, 1, 1}, {32, 1, 1});
  EXPECT_TRUE(count_threadgroups(kernel_dispatch) == 4);
}

void should_cover_a_quarter_of_sixteen_cores_with_four_threadgroups() {
  EXPECT_TRUE(gpu_core_coverage(4, 16) == 0.25);
  EXPECT_TRUE(!can_fill_gpu(4, 16));
}

void should_fill_the_gpu_with_hundreds_of_threadgroups() {
  EXPECT_TRUE(gpu_core_coverage(512, 16) == 1.0);
  EXPECT_TRUE(can_fill_gpu(512, 16));
}

// --- edge / boundary ---

void should_round_a_partial_threadgroup_up_in_every_dimension() {
  const KernelDispatch kernel_dispatch = make_dispatch(DispatchGridUnit::threads, {33, 5, 1}, {32, 4, 1});
  const GridSize formed_grid = threadgroup_grid(kernel_dispatch);
  EXPECT_TRUE(formed_grid.width == 2 && formed_grid.height == 2 && formed_grid.depth == 1);
  EXPECT_TRUE(count_threadgroups(kernel_dispatch) == 4);
}

void should_fill_the_gpu_with_exactly_one_threadgroup_per_core() {
  EXPECT_TRUE(can_fill_gpu(16, 16));
  EXPECT_TRUE(!can_fill_gpu(15, 16));
}

void should_form_no_threadgroups_from_an_empty_grid() {
  EXPECT_TRUE(count_threadgroups(make_dispatch(DispatchGridUnit::threads, {0, 1, 1}, {32, 1, 1})) == 0);
  EXPECT_TRUE(gpu_core_coverage(0, 16) == 0.0);
}

// --- negative ---

void should_form_no_threadgroups_from_a_zero_sized_threadgroup() {
  EXPECT_TRUE(count_threadgroups(make_dispatch(DispatchGridUnit::threads, {128, 1, 1}, {0, 1, 1})) == 0);
  EXPECT_TRUE(count_threadgroups(make_dispatch(DispatchGridUnit::threadgroups, {8, 1, 1}, {1, 0, 1})) == 0);
}

void should_claim_nothing_when_the_core_count_is_unknown() {
  EXPECT_TRUE(gpu_core_coverage(4, 0) < 0.0);
  EXPECT_TRUE(!can_fill_gpu(1000, 0));
  EXPECT_TRUE(!can_fill_gpu(1000, -3));
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_count_the_threadgroups_a_threadgroup_grid_names),
      TEST_CASE(should_form_threadgroups_from_a_thread_grid),
      TEST_CASE(should_cover_a_quarter_of_sixteen_cores_with_four_threadgroups),
      TEST_CASE(should_fill_the_gpu_with_hundreds_of_threadgroups),
      TEST_CASE(should_round_a_partial_threadgroup_up_in_every_dimension),
      TEST_CASE(should_fill_the_gpu_with_exactly_one_threadgroup_per_core),
      TEST_CASE(should_form_no_threadgroups_from_an_empty_grid),
      TEST_CASE(should_form_no_threadgroups_from_a_zero_sized_threadgroup),
      TEST_CASE(should_claim_nothing_when_the_core_count_is_unknown),
  });
}
