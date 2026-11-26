#pragma once

// Qwen3.5 text config: one concern only — the architecture numbers the
// decode step needs, read from a model directory's config.json and checked.

#include <string>
#include <vector>

namespace mlx_perf::qwen35 {

enum class LayerKind { gated_delta_net, full_attention };

struct Qwen35TextConfig {
  int hidden_size = 0;
  int intermediate_size = 0;
  int vocabulary_size = 0;
  float rms_norm_epsilon = 1e-6f;
  // Full (softmax) attention layers.
  int attention_head_count = 0;
  int key_value_head_count = 0;
  int attention_head_dim = 0;
  int rotary_dims = 0;  // head_dim × partial_rotary_factor
  float rope_theta = 0.0f;
  // Gated DeltaNet (linear attention) layers.
  int delta_key_head_count = 0;
  int delta_value_head_count = 0;
  int delta_key_head_dim = 0;
  int delta_value_head_dim = 0;
  int delta_conv_kernel = 0;
  std::vector<LayerKind> layer_kinds;
  // From the top-level "quantization" block; bits are derived per tensor.
  int quantization_group_size = 0;
};

// Throws std::runtime_error naming the missing or invalid field. MoE
// checkpoints (num_experts > 0) are rejected: this decode step is dense.
Qwen35TextConfig load_qwen35_text_config(const std::string& model_directory);

} // namespace mlx_perf::qwen35
