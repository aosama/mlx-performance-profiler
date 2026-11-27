// Integration tests for cuda_gpu_device_facts: the GPU's name and
// multiprocessor count, read from the CUDA driver of the machine running
// the tests.

#include "gpu_device_facts.h"
#include "test_harness.h"

#include <string>

namespace {

// --- positive ---

void should_report_the_gpu_and_its_multiprocessor_count() {
  const mlx_perf::GpuDeviceFacts& gpu_device_facts = mlx_perf::read_gpu_device_facts();
  EXPECT_TRUE(gpu_device_facts.gpu_core_count > 0);
  EXPECT_TRUE(!gpu_device_facts.gpu_name.empty());
}

// --- edge / boundary ---

// NVIDIA GPUs that MLX's CUDA backend supports (compute capability 7.5 and
// later) ship from a handful of multiprocessors to under 200; a figure far
// outside that means the wrong attribute was read.
void should_read_a_multiprocessor_count_within_a_plausible_range() {
  const int multiprocessor_count = mlx_perf::read_gpu_device_facts().gpu_core_count;
  EXPECT_TRUE(multiprocessor_count >= 1 && multiprocessor_count <= 512);
}

// --- negative ---

void should_read_the_facts_once_and_return_the_same_object() {
  EXPECT_TRUE(&mlx_perf::read_gpu_device_facts() == &mlx_perf::read_gpu_device_facts());
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_report_the_gpu_and_its_multiprocessor_count),
      TEST_CASE(should_read_a_multiprocessor_count_within_a_plausible_range),
      TEST_CASE(should_read_the_facts_once_and_return_the_same_object),
  });
}
