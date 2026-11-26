#include "metal_object_classes.h"

#include "mlx/backend/metal/device.h"

namespace mlx_perf {
namespace {

Class class_of(const void* metal_object) {
  return metal_object == nullptr ? nullptr : object_getClass(reinterpret_cast<id>(const_cast<void*>(metal_object)));
}

MetalObjectClasses probe_metal_object_classes() {
  MetalObjectClasses metal_object_classes;
  MTL::Device* metal_device = mlx::core::metal::device(mlx::core::Device::gpu).mtl_device();
  if (metal_device == nullptr) {
    return metal_object_classes;
  }
  metal_object_classes.device_class = class_of(metal_device);
  NS::AutoreleasePool* autorelease_pool = NS::AutoreleasePool::alloc()->init();
  MTL::CommandQueue* probe_queue = metal_device->newCommandQueue();
  if (probe_queue != nullptr) {
    // The same call MLX makes (device.cpp), whose class can differ from a
    // plain commandBuffer's.
    MTL::CommandBuffer* probe_buffer = probe_queue->commandBufferWithUnretainedReferences();
    metal_object_classes.command_buffer_class = class_of(probe_buffer);
    MTL::ComputeCommandEncoder* probe_encoder =
        probe_buffer == nullptr ? nullptr : probe_buffer->computeCommandEncoder(MTL::DispatchTypeConcurrent);
    if (probe_encoder != nullptr) {
      metal_object_classes.compute_encoder_class = class_of(probe_encoder);
      probe_encoder->endEncoding();
    }
    probe_queue->release();
  }
  MTL::SharedEvent* probe_event = metal_device->newSharedEvent();
  if (probe_event != nullptr) {
    metal_object_classes.shared_event_class = class_of(probe_event);
    probe_event->release();
  }
  autorelease_pool->release();
  return metal_object_classes;
}

} // namespace

const MetalObjectClasses& find_metal_object_classes() {
  static const MetalObjectClasses found_classes = probe_metal_object_classes();
  return found_classes;
}

} // namespace mlx_perf
