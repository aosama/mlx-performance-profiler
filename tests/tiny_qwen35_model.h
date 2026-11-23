#pragma once

// A tiny synthetic Qwen3.5 model for tests: random weights quantized by MLX
// itself, one Gated DeltaNet layer and one full-attention layer. No
// checkpoint files needed. Shared by every test that needs real decode math.

#include "qwen35/qwen35_config.h"
#include "qwen35/qwen35_weights.h"

#include "mlx/mlx.h"

namespace tiny_qwen35_model {

using namespace mlx::core;
using namespace mlx_plan::qwen35;

inline constexpr int TINY_VOCABULARY = 64;

inline Qwen35TextConfig make_tiny_config() {
  Qwen35TextConfig config;
  config.hidden_size = 128;
  config.intermediate_size = 256;
  config.vocabulary_size = TINY_VOCABULARY;
  config.attention_head_count = 2;
  config.key_value_head_count = 1;
  config.attention_head_dim = 64;
  config.rotary_dims = 16;
  config.rope_theta = 10000.0f;
  config.delta_key_head_count = 2;
  config.delta_value_head_count = 2;
  config.delta_key_head_dim = 64;
  config.delta_value_head_dim = 64;
  config.delta_conv_kernel = 4;
  config.layer_kinds = {LayerKind::gated_delta_net, LayerKind::full_attention};
  config.quantization_group_size = 64;
  return config;
}

inline QuantizedLinear make_linear(int output_features, int input_features, int bits = 4) {
  const array dense_weight =
      astype(multiply(random::normal({output_features, input_features}), array(0.05f)),
             bfloat16);
  const auto quantized = quantize(dense_weight, 64, bits);
  return {quantized[0], quantized[1], quantized[2], 64, bits};
}

inline array norm_weight(int width) { return ones({width}, bfloat16); }

inline DecoderLayerWeights make_layer(const Qwen35TextConfig& config, LayerKind layer_kind) {
  DecoderLayerWeights layer{layer_kind, norm_weight(config.hidden_size),
                            norm_weight(config.hidden_size), std::nullopt, std::nullopt,
                            make_linear(config.intermediate_size, config.hidden_size),
                            make_linear(config.intermediate_size, config.hidden_size),
                            make_linear(config.hidden_size, config.intermediate_size)};
  const int value_width = config.delta_value_head_count * config.delta_value_head_dim;
  const int conv_dim = 2 * config.delta_key_head_count * config.delta_key_head_dim + value_width;
  if (layer_kind == LayerKind::gated_delta_net) {
    layer.delta_net = GatedDeltaNetWeights{
        make_linear(conv_dim, config.hidden_size),
        make_linear(value_width, config.hidden_size),
        make_linear(config.delta_value_head_count, config.hidden_size),
        make_linear(config.delta_value_head_count, config.hidden_size),
        astype(multiply(random::normal({conv_dim, config.delta_conv_kernel, 1}), array(0.1f)),
               bfloat16),
        zeros({config.delta_value_head_count}, float32),
        ones({config.delta_value_head_count}, bfloat16),
        norm_weight(config.delta_value_head_dim),
        make_linear(config.hidden_size, value_width)};
  } else {
    const int head_width = config.attention_head_count * config.attention_head_dim;
    const int key_width = config.key_value_head_count * config.attention_head_dim;
    layer.attention = FullAttentionWeights{
        make_linear(2 * head_width, config.hidden_size), make_linear(key_width, config.hidden_size),
        make_linear(key_width, config.hidden_size), make_linear(config.hidden_size, head_width),
        norm_weight(config.attention_head_dim), norm_weight(config.attention_head_dim)};
  }
  return layer;
}

inline Qwen35Weights make_tiny_weights(const Qwen35TextConfig& config) {
  Qwen35Weights weights{make_linear(TINY_VOCABULARY, config.hidden_size), {},
                        norm_weight(config.hidden_size)};
  for (const LayerKind layer_kind : config.layer_kinds) {
    weights.layers.push_back(make_layer(config, layer_kind));
  }
  return weights;
}

} // namespace tiny_qwen35_model
