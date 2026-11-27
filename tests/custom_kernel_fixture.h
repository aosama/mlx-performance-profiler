#pragma once

// Custom kernel fixture: one concern only — the custom GPU kernels the
// shared tests launch, written in the language of the GPU backend this
// build links (Metal Shading Language, or CUDA C++), so one test checks one
// behaviour on either backend.

#include "mlx/fast.h"

#include <string>

namespace custom_kernel_fixture {

// Thread i doubles element i of `input_values` into `output_values`.
inline mlx::core::fast::CustomKernelFunction make_doubling_kernel(const std::string& author_kernel_name) {
#if defined(MLX_PERF_GPU_BACKEND_CUDA)
  return mlx::core::fast::cuda_kernel(
      author_kernel_name, {"input_values"}, {"output_values"},
      "unsigned int element_index = blockIdx.x * blockDim.x + threadIdx.x;\n"
      "output_values[element_index] = 2.0f * input_values[element_index];\n");
#else
  return mlx::core::fast::metal_kernel(
      author_kernel_name, {"input_values"}, {"output_values"},
      "uint element_index = thread_position_in_grid.x;\n"
      "output_values[element_index] = 2.0f * input_values[element_index];\n");
#endif
}

// Every thread doubles each grid-stride element, so any launch size does the
// same total work and only the number of threadgroups (blocks) differs.
inline mlx::core::fast::CustomKernelFunction make_grid_stride_doubling_kernel(
    const std::string& author_kernel_name) {
#if defined(MLX_PERF_GPU_BACKEND_CUDA)
  return mlx::core::fast::cuda_kernel(
      author_kernel_name, {"input_values"}, {"output_values"},
      "for (unsigned int element_index = blockIdx.x * blockDim.x + threadIdx.x;\n"
      "     element_index < input_values_shape[0]; element_index += gridDim.x * blockDim.x) {\n"
      "  output_values[element_index] = 2.0f * input_values[element_index];\n"
      "}\n");
#else
  return mlx::core::fast::metal_kernel(
      author_kernel_name, {"input_values"}, {"output_values"},
      "for (uint element_index = thread_position_in_grid.x; element_index < input_values_shape[0];\n"
      "     element_index += threads_per_grid.x) {\n"
      "  output_values[element_index] = 2.0f * input_values[element_index];\n"
      "}\n");
#endif
}

} // namespace custom_kernel_fixture
