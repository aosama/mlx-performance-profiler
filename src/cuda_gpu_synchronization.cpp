#include "cuda_gpu_synchronization.h"

#include <cuda.h>

namespace mlx_perf {

void wait_for_every_gpu_submission() {
  CUcontext current_context = nullptr;
  if (cuCtxGetCurrent(&current_context) == CUDA_SUCCESS && current_context != nullptr) {
    cuCtxSynchronize();
    return;
  }
  CUdevice cuda_device = 0;
  CUcontext primary_context = nullptr;
  if (cuDeviceGet(&cuda_device, 0) != CUDA_SUCCESS ||
      cuDevicePrimaryCtxRetain(&primary_context, cuda_device) != CUDA_SUCCESS) {
    return;
  }
  if (cuCtxPushCurrent(primary_context) == CUDA_SUCCESS) {
    cuCtxSynchronize();
    cuCtxPopCurrent(&current_context);
  }
  cuDevicePrimaryCtxRelease(cuda_device);
}

} // namespace mlx_perf
