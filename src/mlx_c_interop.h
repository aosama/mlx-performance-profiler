#pragma once

// MLX-C interop: one concern only — read MLX-C array handles as C++
// mlx::core::array values. This is the single place that relies on MLX-C's
// private layout (an mlx_array's ctx is a heap mlx::core::array*, as
// MLX-C's own private/array.h defines it); a test pins that assumption.

#include "mlx/array.h"
#include "mlx/c/vector.h"

#include <vector>

namespace mlx_plan {

// Copies (shares) every array in `vector_handle`. Throws std::runtime_error
// on an MLX-C failure or a null handle.
std::vector<mlx::core::array> read_mlx_c_vector(mlx_vector_array vector_handle);

} // namespace mlx_plan
