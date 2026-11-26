#include "mlx_c_interop.h"

#include "mlx/c/array.h"

#include <stdexcept>
#include <string>

namespace mlx_perf {

std::vector<mlx::core::array> read_mlx_c_vector(mlx_vector_array vector_handle) {
  if (vector_handle.ctx == nullptr) {
    throw std::runtime_error("mlx_vector_array handle is null");
  }
  const std::size_t element_count = mlx_vector_array_size(vector_handle);
  std::vector<mlx::core::array> core_arrays;
  core_arrays.reserve(element_count);
  for (std::size_t element_index = 0; element_index < element_count; ++element_index) {
    mlx_array element_handle = mlx_array_new();
    if (mlx_vector_array_get(&element_handle, vector_handle, element_index) != 0 ||
        element_handle.ctx == nullptr) {
      mlx_array_free(element_handle);
      throw std::runtime_error("cannot read element " + std::to_string(element_index) +
                               " of an mlx_vector_array");
    }
    // Copying an mlx::core::array shares its descriptor, so the C++ value
    // stays valid after the MLX-C handle is freed.
    core_arrays.push_back(*static_cast<mlx::core::array*>(element_handle.ctx));
    mlx_array_free(element_handle);
  }
  return core_arrays;
}

} // namespace mlx_perf
