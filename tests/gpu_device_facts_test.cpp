// Integration tests for gpu_device_facts: the GPU's name and core count, read
// from Metal and the IO registry of the Mac running the tests.

#include "gpu_device_facts.h"
#include "test_harness.h"

#include <string>

namespace {

// --- positive ---

void should_report_the_gpu_and_its_core_count() {
  const mlx_perf::GpuDeviceFacts& gpu_device_facts = mlx_perf::read_gpu_device_facts();
  EXPECT_TRUE(gpu_device_facts.gpu_core_count > 0);
  EXPECT_TRUE(gpu_device_facts.gpu_name.find("Apple") != std::string::npos);
}

// --- edge / boundary ---

// Apple silicon ships 7 to 80 GPU cores; a figure outside that range means
// the wrong registry property was read.
void should_read_a_core_count_within_apple_silicons_range() {
  const int gpu_core_count = mlx_perf::read_gpu_device_facts().gpu_core_count;
  EXPECT_TRUE(gpu_core_count >= 7 && gpu_core_count <= 80);
}

// --- negative ---

void should_read_the_facts_once_and_return_the_same_object() {
  EXPECT_TRUE(&mlx_perf::read_gpu_device_facts() == &mlx_perf::read_gpu_device_facts());
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_report_the_gpu_and_its_core_count),
      TEST_CASE(should_read_a_core_count_within_apple_silicons_range),
      TEST_CASE(should_read_the_facts_once_and_return_the_same_object),
  });
}
