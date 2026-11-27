// Integration tests for machine_ceilings: the reading and arithmetic
// ceilings measured on this machine's GPU with MLX's own kernels.

#include "machine_ceilings.h"
#include "test_harness.h"

#include "mlx/memory.h"

namespace {

using namespace mlx_perf;

// Wide bounds on purpose: they hold from the smallest Apple GPU to a large
// NVIDIA one, and still catch a unit slip (bits for bytes, microseconds for
// seconds), which lands orders of magnitude outside.
constexpr double LOWEST_PLAUSIBLE_READING_BYTES_PER_SECOND = 10e9;
constexpr double HIGHEST_PLAUSIBLE_READING_BYTES_PER_SECOND = 10e12;
constexpr double LOWEST_PLAUSIBLE_OPERATIONS_PER_SECOND = 100e9;
constexpr double HIGHEST_PLAUSIBLE_OPERATIONS_PER_SECOND = 10e15;

// --- positive ---

void should_measure_both_ceilings_on_this_gpu() {
  const MachineCeilings& machine_ceilings = read_machine_ceilings();
  EXPECT_TRUE(machine_ceilings.is_measured);
  EXPECT_TRUE(machine_ceilings.arithmetic_dtype == "float16" ||
              machine_ceilings.arithmetic_dtype == "bfloat16");
}

// --- edge / boundary ---

void should_land_within_physically_plausible_bounds() {
  const MachineCeilings& machine_ceilings = read_machine_ceilings();
  EXPECT_TRUE(machine_ceilings.memory_read_bytes_per_second > LOWEST_PLAUSIBLE_READING_BYTES_PER_SECOND);
  EXPECT_TRUE(machine_ceilings.memory_read_bytes_per_second < HIGHEST_PLAUSIBLE_READING_BYTES_PER_SECOND);
  EXPECT_TRUE(machine_ceilings.arithmetic_operations_per_second > LOWEST_PLAUSIBLE_OPERATIONS_PER_SECOND);
  EXPECT_TRUE(machine_ceilings.arithmetic_operations_per_second < HIGHEST_PLAUSIBLE_OPERATIONS_PER_SECOND);
}

void should_calibrate_once_per_process() {
  const MachineCeilings& first_reading = read_machine_ceilings();
  const MachineCeilings& second_reading = read_machine_ceilings();
  EXPECT_TRUE(&first_reading == &second_reading);
}

// --- negative ---

void should_leave_no_calibration_memory_in_mlx_pool() {
  read_machine_ceilings();
  EXPECT_TRUE(mlx::core::get_cache_memory() < 64 * 1024 * 1024);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_measure_both_ceilings_on_this_gpu),
      TEST_CASE(should_land_within_physically_plausible_bounds),
      TEST_CASE(should_calibrate_once_per_process),
      TEST_CASE(should_leave_no_calibration_memory_in_mlx_pool),
  });
}
