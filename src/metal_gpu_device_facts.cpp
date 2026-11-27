#include "gpu_device_facts.h"

#include "mlx/backend/metal/device.h"

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

namespace mlx_perf {
namespace {

// "AGXAccelerator" matches every Apple-GPU accelerator subclass
// (AGXAcceleratorG17X on an M5, for example), so no chip list is needed.
int read_gpu_core_count_from_io_registry() {
  const io_service_t accelerator_service =
      IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"));
  if (accelerator_service == IO_OBJECT_NULL) {
    return 0;
  }
  int gpu_core_count = 0;
  const CFTypeRef core_count_property = IORegistryEntryCreateCFProperty(
      accelerator_service, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
  if (core_count_property != nullptr) {
    if (CFGetTypeID(core_count_property) == CFNumberGetTypeID()) {
      CFNumberGetValue(static_cast<CFNumberRef>(core_count_property), kCFNumberIntType,
                       &gpu_core_count);
    }
    CFRelease(core_count_property);
  }
  IOObjectRelease(accelerator_service);
  return gpu_core_count > 0 ? gpu_core_count : 0;
}

std::string read_gpu_name() {
  MTL::Device* metal_device = mlx::core::metal::device(mlx::core::Device::gpu).mtl_device();
  if (metal_device == nullptr || metal_device->name() == nullptr) {
    return std::string();
  }
  return metal_device->name()->utf8String();
}

GpuDeviceFacts query_gpu_device_facts() {
  GpuDeviceFacts gpu_device_facts;
  gpu_device_facts.gpu_name = read_gpu_name();
  gpu_device_facts.gpu_core_count = read_gpu_core_count_from_io_registry();
  return gpu_device_facts;
}

} // namespace

const GpuDeviceFacts& read_gpu_device_facts() {
  static const GpuDeviceFacts cached_gpu_device_facts = query_gpu_device_facts();
  return cached_gpu_device_facts;
}

} // namespace mlx_perf
