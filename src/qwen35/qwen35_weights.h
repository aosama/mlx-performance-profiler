#pragma once

// Qwen3.5 weights: one concern only — load a dense Qwen3.5 checkpoint's
// language-model tensors from safetensors into per-layer structs. MLX maps
// the file lazily, so nothing is read from disk until a tensor is used.

#include "qwen35_config.h"

#include "mlx/array.h"

#include <optional>
#include <string>
#include <vector>

namespace mlx_perf::qwen35 {

// An MLX affine-quantized linear layer (weight packed into uint32).
struct QuantizedLinear {
  mlx::core::array packed_weight;
  mlx::core::array scales;
  mlx::core::array biases;
  int group_size = 64;
  int bits = 4;
};

struct GatedDeltaNetWeights {
  QuantizedLinear qkv_projection;
  QuantizedLinear gate_projection;   // in_proj_z
  QuantizedLinear beta_projection;   // in_proj_b
  QuantizedLinear decay_projection;  // in_proj_a
  mlx::core::array conv_weight;      // [conv_dim, kernel, 1], depthwise
  mlx::core::array log_decay_rate;   // A_log
  mlx::core::array decay_time_bias;  // dt_bias
  mlx::core::array output_norm_weight;
  QuantizedLinear output_projection;
};

struct FullAttentionWeights {
  QuantizedLinear query_projection;  // queries and output gate, interleaved per head
  QuantizedLinear key_projection;
  QuantizedLinear value_projection;
  QuantizedLinear output_projection;
  mlx::core::array query_norm_weight;
  mlx::core::array key_norm_weight;
};

struct DecoderLayerWeights {
  LayerKind layer_kind = LayerKind::gated_delta_net;
  mlx::core::array input_norm_weight;
  mlx::core::array post_attention_norm_weight;
  std::optional<GatedDeltaNetWeights> delta_net;
  std::optional<FullAttentionWeights> attention;
  QuantizedLinear mlp_gate;
  QuantizedLinear mlp_up;
  QuantizedLinear mlp_down;
};

struct Qwen35Weights {
  // The embedding doubles as the LM head (tie_word_embeddings).
  QuantizedLinear token_embedding;
  std::vector<DecoderLayerWeights> layers;
  mlx::core::array final_norm_weight;
};

// Throws std::runtime_error naming the first missing tensor.
Qwen35Weights load_qwen35_weights(const std::string& model_directory,
                                  const Qwen35TextConfig& config);

// Reads every tensor into memory now (with progress). load_safetensors is
// lazy, so without this the first measured graph would include disk reads.
void materialize_qwen35_weights(const Qwen35Weights& weights);

// Bits per weight, derived from packed and scale shapes rather than trusted
// from config: mixed-precision checkpoints quantize layers differently.
int derive_quantization_bits(const mlx::core::array& packed_weight,
                             const mlx::core::array& scales, int group_size);

} // namespace mlx_perf::qwen35
