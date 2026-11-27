#include "gpu_device_facts.h"

#include <cuda.h>

#include <array>

namespace mlx_perf {
namespace {

// Device 0 is the one MLX's default GPU stream runs on (mlx::core::Device::gpu
// has index 0 unless the caller chose another).
constexpr int MLX_DEFAULT_GPU_ORDINAL = 0;
constexpr int GPU_NAME_CAPACITY = 256;

// The streaming multiprocessor count plays the role a Mac's GPU core count
// plays in dispatch_occupancy: a thread block runs on one multiprocessor, so
// a launch with fewer blocks than multiprocessors leaves some idle. The
// driver reports it exactly, as CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
GpuDeviceFacts query_gpu_device_facts() {
  GpuDeviceFacts gpu_device_facts;
  CUdevice cuda_device = 0;
  if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&cuda_device, MLX_DEFAULT_GPU_ORDINAL) != CUDA_SUCCESS) {
    return gpu_device_facts;
  }
  std::array<char, GPU_NAME_CAPACITY> gpu_name_characters{};
  if (cuDeviceGetName(gpu_name_characters.data(), GPU_NAME_CAPACITY, cuda_device) == CUDA_SUCCESS) {
    gpu_device_facts.gpu_name = gpu_name_characters.data();
  }
  int multiprocessor_count = 0;
  if (cuDeviceGetAttribute(&multiprocessor_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuda_device) ==
          CUDA_SUCCESS &&
      multiprocessor_count > 0) {
    gpu_device_facts.gpu_core_count = multiprocessor_count;
  }
  return gpu_device_facts;
}

} // namespace

const GpuDeviceFacts& read_gpu_device_facts() {
  static const GpuDeviceFacts cached_gpu_device_facts = query_gpu_device_facts();
  return cached_gpu_device_facts;
}

} // namespace mlx_perf
