#include "qwen35_weights.h"

#include "json_value.h"
#include "progress_reporter.h"

#include "mlx/io.h"
#include "mlx/transforms.h"

#include <filesystem>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace mlx_plan::qwen35 {
namespace {

using mlx::core::array;
using TensorMap = std::unordered_map<std::string, array>;

// Sharded checkpoints list their files in model.safetensors.index.json;
// single-file ones have just model.safetensors.
std::vector<std::string> list_safetensors_files(const std::string& model_directory) {
  const std::string index_path = model_directory + "/model.safetensors.index.json";
  if (!std::filesystem::exists(index_path)) {
    return {model_directory + "/model.safetensors"};
  }
  // Bound to a local: in a range-for, a temporary document would be
  // destroyed before the loop body reads members through it.
  const JsonValue shard_index = parse_json_file(index_path);
  std::set<std::string> shard_names;
  for (const auto& [tensor_name, shard_name] : shard_index.at("weight_map").as_object()) {
    shard_names.insert(shard_name.as_string());
  }
  std::vector<std::string> shard_paths;
  for (const std::string& shard_name : shard_names) {
    shard_paths.push_back(model_directory + "/" + shard_name);
  }
  return shard_paths;
}

TensorMap load_all_tensors(const std::string& model_directory) {
  TensorMap tensor_name_to_array;
  for (const std::string& shard_path : list_safetensors_files(model_directory)) {
    for (auto& [tensor_name, tensor] : mlx::core::load_safetensors(shard_path).first) {
      tensor_name_to_array.insert_or_assign(tensor_name, tensor);
    }
  }
  return tensor_name_to_array;
}

class TensorLookup {
 public:
  TensorLookup(TensorMap tensor_name_to_array, std::string name_prefix, int group_size)
      : tensor_name_to_array_(std::move(tensor_name_to_array)),
        name_prefix_(std::move(name_prefix)),
        group_size_(group_size) {}

  array tensor(const std::string& relative_name) const {
    const auto tensor_entry = tensor_name_to_array_.find(name_prefix_ + relative_name);
    if (tensor_entry == tensor_name_to_array_.end()) {
      throw std::runtime_error("checkpoint has no tensor " + name_prefix_ + relative_name);
    }
    return tensor_entry->second;
  }

  QuantizedLinear linear(const std::string& relative_name) const {
    QuantizedLinear quantized_linear{tensor(relative_name + ".weight"),
                                     tensor(relative_name + ".scales"),
                                     tensor(relative_name + ".biases"), group_size_, 0};
    quantized_linear.bits = derive_quantization_bits(
        quantized_linear.packed_weight, quantized_linear.scales, group_size_);
    return quantized_linear;
  }

 private:
  TensorMap tensor_name_to_array_;
  std::string name_prefix_;
  int group_size_;
};

// Vision-language checkpoints prefix the text model; text-only ones don't.
std::string detect_name_prefix(const TensorMap& tensor_name_to_array) {
  for (const char* candidate_prefix : {"language_model.model.", "model."}) {
    if (tensor_name_to_array.count(std::string(candidate_prefix) + "norm.weight") != 0) {
      return candidate_prefix;
    }
  }
  throw std::runtime_error("cannot find the language model's norm.weight");
}

GatedDeltaNetWeights load_delta_net(const TensorLookup& tensors, const std::string& layer) {
  const std::string module = layer + "linear_attn.";
  return {tensors.linear(module + "in_proj_qkv"),  tensors.linear(module + "in_proj_z"),
          tensors.linear(module + "in_proj_b"),    tensors.linear(module + "in_proj_a"),
          tensors.tensor(module + "conv1d.weight"), tensors.tensor(module + "A_log"),
          tensors.tensor(module + "dt_bias"),       tensors.tensor(module + "norm.weight"),
          tensors.linear(module + "out_proj")};
}

FullAttentionWeights load_attention(const TensorLookup& tensors, const std::string& layer) {
  const std::string module = layer + "self_attn.";
  return {tensors.linear(module + "q_proj"),        tensors.linear(module + "k_proj"),
          tensors.linear(module + "v_proj"),        tensors.linear(module + "o_proj"),
          tensors.tensor(module + "q_norm.weight"), tensors.tensor(module + "k_norm.weight")};
}

DecoderLayerWeights load_layer(const TensorLookup& tensors, int layer_index,
                               LayerKind layer_kind) {
  const std::string layer = "layers." + std::to_string(layer_index) + ".";
  DecoderLayerWeights layer_weights{
      layer_kind,
      tensors.tensor(layer + "input_layernorm.weight"),
      tensors.tensor(layer + "post_attention_layernorm.weight"),
      std::nullopt,
      std::nullopt,
      tensors.linear(layer + "mlp.gate_proj"),
      tensors.linear(layer + "mlp.up_proj"),
      tensors.linear(layer + "mlp.down_proj")};
  if (layer_kind == LayerKind::gated_delta_net) {
    layer_weights.delta_net = load_delta_net(tensors, layer);
  } else {
    layer_weights.attention = load_attention(tensors, layer);
  }
  return layer_weights;
}

void append_linear(const QuantizedLinear& linear, std::vector<array>& layer_tensors) {
  layer_tensors.insert(layer_tensors.end(),
                       {linear.packed_weight, linear.scales, linear.biases});
}

std::vector<array> collect_layer_tensors(const DecoderLayerWeights& layer) {
  std::vector<array> layer_tensors{layer.input_norm_weight, layer.post_attention_norm_weight};
  for (const QuantizedLinear* linear : {&layer.mlp_gate, &layer.mlp_up, &layer.mlp_down}) {
    append_linear(*linear, layer_tensors);
  }
  if (layer.delta_net) {
    const GatedDeltaNetWeights& delta_net = *layer.delta_net;
    for (const QuantizedLinear* linear :
         {&delta_net.qkv_projection, &delta_net.gate_projection, &delta_net.beta_projection,
          &delta_net.decay_projection, &delta_net.output_projection}) {
      append_linear(*linear, layer_tensors);
    }
    layer_tensors.insert(layer_tensors.end(),
                         {delta_net.conv_weight, delta_net.log_decay_rate,
                          delta_net.decay_time_bias, delta_net.output_norm_weight});
  }
  if (layer.attention) {
    const FullAttentionWeights& attention = *layer.attention;
    for (const QuantizedLinear* linear :
         {&attention.query_projection, &attention.key_projection,
          &attention.value_projection, &attention.output_projection}) {
      append_linear(*linear, layer_tensors);
    }
    layer_tensors.insert(layer_tensors.end(),
                         {attention.query_norm_weight, attention.key_norm_weight});
  }
  return layer_tensors;
}

} // namespace

void materialize_qwen35_weights(const Qwen35Weights& weights) {
  const int unit_count = static_cast<int>(weights.layers.size()) + 1;
  ProgressReporter progress("loading weights (layers)", unit_count);
  std::vector<array> shared_tensors{weights.final_norm_weight};
  append_linear(weights.token_embedding, shared_tensors);
  mlx::core::eval(shared_tensors);
  for (std::size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
    mlx::core::eval(collect_layer_tensors(weights.layers[layer_index]));
    progress.update(static_cast<int>(layer_index) + 2);
  }
  progress.finish(unit_count);
}

int derive_quantization_bits(const array& packed_weight, const array& scales,
                             int group_size) {
  if (packed_weight.ndim() < 2 || scales.ndim() < 2 || group_size <= 0) {
    throw std::runtime_error("quantized tensor has an unexpected shape");
  }
  const long long input_features =
      static_cast<long long>(scales.shape(-1)) * group_size;
  const long long packed_bits = static_cast<long long>(packed_weight.shape(-1)) * 32;
  if (input_features == 0 || packed_bits % input_features != 0) {
    throw std::runtime_error("packed weight width is not a whole number of bits");
  }
  return static_cast<int>(packed_bits / input_features);
}

Qwen35Weights load_qwen35_weights(const std::string& model_directory,
                                  const Qwen35TextConfig& config) {
  TensorMap tensor_name_to_array = load_all_tensors(model_directory);
  std::string name_prefix = detect_name_prefix(tensor_name_to_array);
  const TensorLookup tensors(std::move(tensor_name_to_array), std::move(name_prefix),
                             config.quantization_group_size);
  Qwen35Weights weights{tensors.linear("embed_tokens"), {}, tensors.tensor("norm.weight")};
  for (int layer_index = 0; layer_index < static_cast<int>(config.layer_kinds.size());
       ++layer_index) {
    weights.layers.push_back(
        load_layer(tensors, layer_index, config.layer_kinds[layer_index]));
  }
  return weights;
}

} // namespace mlx_plan::qwen35
