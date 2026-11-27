#include "cuda_mlx_streams.h"

#include "mlx/backend/cuda/device.h"

namespace mlx_perf {

// Each MLX GPU stream owns one CommandEncoder, created on first use, whose
// CUDA stream every graph launch for that MLX stream goes to
// (cu::CommandEncoder::commit). cudaStream_t and CUstream are one type.
CUstream cuda_stream_of(const mlx::core::Stream& mlx_stream) {
  if (mlx_stream.device != mlx::core::Device::gpu) {
    return nullptr;
  }
  const cudaStream_t encoder_stream = mlx::core::cu::get_command_encoder(mlx_stream).stream();
  return static_cast<CUstream>(encoder_stream);
}

} // namespace mlx_perf
