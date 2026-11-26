// Integration tests for metal_object_classes on the Metal GPU: the classes
// found must be the ones MLX's own objects have, and must answer the methods
// the recorders hook.

#include "metal_object_classes.h"
#include "test_harness.h"

#include "mlx/backend/metal/device.h"
#include "mlx/mlx.h"

#include <objc/runtime.h>

namespace {

using mlx_perf::MetalObjectClasses;

bool responds_to(Class metal_class, const char* selector_name) {
  return metal_class != nullptr && class_getInstanceMethod(metal_class, sel_registerName(selector_name)) != nullptr;
}

// --- positive ---

void should_find_every_class_on_this_gpu() {
  const MetalObjectClasses& metal_object_classes = mlx_perf::find_metal_object_classes();
  EXPECT_TRUE(metal_object_classes.device_class != nullptr);
  EXPECT_TRUE(metal_object_classes.command_buffer_class != nullptr);
  EXPECT_TRUE(metal_object_classes.compute_encoder_class != nullptr);
  EXPECT_TRUE(metal_object_classes.shared_event_class != nullptr);
}

void should_find_the_class_of_mlxs_own_device() {
  MTL::Device* metal_device = mlx::core::metal::device(mlx::core::Device::gpu).mtl_device();
  EXPECT_TRUE(mlx_perf::find_metal_object_classes().device_class ==
              object_getClass(reinterpret_cast<id>(metal_device)));
}

void should_find_classes_that_answer_the_hooked_methods() {
  const MetalObjectClasses& metal_object_classes = mlx_perf::find_metal_object_classes();
  EXPECT_TRUE(responds_to(metal_object_classes.command_buffer_class, "commit"));
  EXPECT_TRUE(responds_to(metal_object_classes.command_buffer_class, "waitUntilCompleted"));
  EXPECT_TRUE(responds_to(metal_object_classes.shared_event_class, "waitUntilSignaledValue:timeoutMS:"));
  EXPECT_TRUE(responds_to(metal_object_classes.compute_encoder_class, "dispatchThreadgroups:threadsPerThreadgroup:"));
}

// --- edge / boundary ---

void should_probe_once_and_return_the_same_classes() {
  const MetalObjectClasses& first_lookup = mlx_perf::find_metal_object_classes();
  const MetalObjectClasses& second_lookup = mlx_perf::find_metal_object_classes();
  EXPECT_TRUE(&first_lookup == &second_lookup);
}

// --- negative ---

void should_not_confuse_one_kind_of_object_for_another() {
  const MetalObjectClasses& metal_object_classes = mlx_perf::find_metal_object_classes();
  EXPECT_TRUE(metal_object_classes.command_buffer_class != metal_object_classes.compute_encoder_class);
  EXPECT_TRUE(!responds_to(metal_object_classes.shared_event_class, "commit"));
  EXPECT_TRUE(!responds_to(metal_object_classes.command_buffer_class, "waitUntilSignaledValue:timeoutMS:"));
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_find_every_class_on_this_gpu),
      TEST_CASE(should_find_the_class_of_mlxs_own_device),
      TEST_CASE(should_find_classes_that_answer_the_hooked_methods),
      TEST_CASE(should_probe_once_and_return_the_same_classes),
      TEST_CASE(should_not_confuse_one_kind_of_object_for_another),
  });
}
