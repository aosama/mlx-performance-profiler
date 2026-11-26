#include "qwen35_config.h"

#include "json_value.h"

#include <stdexcept>

namespace mlx_perf::qwen35 {
namespace {

int read_positive_int(const JsonValue& text_config, const std::string& field_name) {
  const double field_value = text_config.at(field_name).as_number();
  if (field_value <= 0 || field_value != static_cast<int>(field_value)) {
    throw std::runtime_error("config field " + field_name + " must be a positive integer");
  }
  return static_cast<int>(field_value);
}

// rope_theta / partial_rotary_factor moved into rope_parameters in newer
// checkpoints; older ones keep them at the top of text_config.
double read_rope_field(const JsonValue& text_config, const std::string& field_name) {
  if (const JsonValue* rope_parameters = text_config.find("rope_parameters")) {
    if (const JsonValue* field = rope_parameters->find(field_name)) {
      return field->as_number();
    }
  }
  return text_config.at(field_name).as_number();
}

std::vector<LayerKind> read_layer_kinds(const JsonValue& text_config) {
  std::vector<LayerKind> layer_kinds;
  for (const JsonValue& layer_type : text_config.at("layer_types").as_array()) {
    if (layer_type.as_string() == "linear_attention") {
      layer_kinds.push_back(LayerKind::gated_delta_net);
    } else if (layer_type.as_string() == "full_attention") {
      layer_kinds.push_back(LayerKind::full_attention);
    } else {
      throw std::runtime_error("unknown layer type " + layer_type.as_string());
    }
  }
  return layer_kinds;
}

void reject_mixture_of_experts(const JsonValue& text_config) {
  const JsonValue* expert_count = text_config.find("num_experts");
  if (expert_count != nullptr && expert_count->is_number() && expert_count->as_number() > 0) {
    throw std::runtime_error("mixture-of-experts checkpoints are not supported yet");
  }
}

} // namespace

Qwen35TextConfig load_qwen35_text_config(const std::string& model_directory) {
  const JsonValue model_config = parse_json_file(model_directory + "/config.json");
  // Vision-language checkpoints nest the language model under text_config.
  const JsonValue* nested_text_config = model_config.find("text_config");
  const JsonValue& text_config = nested_text_config ? *nested_text_config : model_config;
  reject_mixture_of_experts(text_config);

  Qwen35TextConfig config;
  config.hidden_size = read_positive_int(text_config, "hidden_size");
  config.intermediate_size = read_positive_int(text_config, "intermediate_size");
  config.vocabulary_size = read_positive_int(text_config, "vocab_size");
  config.rms_norm_epsilon = static_cast<float>(text_config.at("rms_norm_eps").as_number());
  config.attention_head_count = read_positive_int(text_config, "num_attention_heads");
  config.key_value_head_count = read_positive_int(text_config, "num_key_value_heads");
  config.attention_head_dim = read_positive_int(text_config, "head_dim");
  config.rotary_dims = static_cast<int>(
      config.attention_head_dim * read_rope_field(text_config, "partial_rotary_factor"));
  config.rope_theta = static_cast<float>(read_rope_field(text_config, "rope_theta"));
  config.delta_key_head_count = read_positive_int(text_config, "linear_num_key_heads");
  config.delta_value_head_count = read_positive_int(text_config, "linear_num_value_heads");
  config.delta_key_head_dim = read_positive_int(text_config, "linear_key_head_dim");
  config.delta_value_head_dim = read_positive_int(text_config, "linear_value_head_dim");
  config.delta_conv_kernel = read_positive_int(text_config, "linear_conv_kernel_dim");
  config.layer_kinds = read_layer_kinds(text_config);
  config.quantization_group_size =
      read_positive_int(model_config.at("quantization"), "group_size");
  if (config.layer_kinds.empty()) {
    throw std::runtime_error("config lists no layers");
  }
  return config;
}

} // namespace mlx_perf::qwen35
