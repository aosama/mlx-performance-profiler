#pragma once

// CUDA MLX streams: one concern only — name the CUDA stream MLX's CUDA
// backend submits a given MLX stream's work to, so CUDA observers can tell
// which MLX stream a driver call came from.
//
// Kept apart because it is the only module that reads MLX's CUDA backend
// internals (mlx/backend/cuda/device.h, the stream's CommandEncoder): if a
// later MLX release changes that surface, this file is the one to adapt.

#include "mlx/stream.h"

#include <cuda.h>

namespace mlx_perf {

// The CUDA stream behind `mlx_stream`, or nullptr for a CPU stream.
CUstream cuda_stream_of(const mlx::core::Stream& mlx_stream);

} // namespace mlx_perf
