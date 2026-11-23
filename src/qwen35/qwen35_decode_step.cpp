#include "qwen35_decode_step.h"

#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/random.h"

#include <cmath>
#include <stdexcept>

namespace mlx_plan::qwen35 {
namespace {

using namespace mlx::core;

constexpr Dtype ACTIVATION_DTYPE = bfloat16;
constexpr int ARRAYS_PER_LAYER_CACHE = 2;

array apply_quantized_linear(const array& input, const QuantizedLinear& linear) {
  return quantized_matmul(input, linear.packed_weight, linear.scales, linear.biases,
                          /*transpose=*/true, linear.group_size, linear.bits);
}

array silu(const array& input) { return multiply(input, sigmoid(input)); }

// Quantized embedding lookup: gather the packed rows, then dequantize only
// those, instead of dequantizing the whole vocabulary table.
array embed_tokens(const QuantizedLinear& embedding, const array& token_ids) {
  const array token_rows = take(embedding.packed_weight, token_ids, 0);
  const array row_scales = take(embedding.scales, token_ids, 0);
  const array row_biases = take(embedding.biases, token_ids, 0);
  return dequantize(token_rows, row_scales, row_biases, embedding.group_size,
                    embedding.bits);
}

int delta_conv_dim(const Qwen35TextConfig& config) {
  return 2 * config.delta_key_head_count * config.delta_key_head_dim +
      config.delta_value_head_count * config.delta_value_head_dim;
}

// mlx-lm's normalize_qk: L2-normalize q and k (as RMS norm with a rescaled
// eps) and fold the Dk^-0.5 readout scale into q.
std::pair<array, array> normalize_query_key(const array& queries, const array& keys,
                                            int key_head_dim) {
  const float inverse_scale = 1.0f / std::sqrt(static_cast<float>(key_head_dim));
  const float rms_epsilon = 1e-6f * inverse_scale * inverse_scale;
  // Scalars carry the activation dtype: Python's weakly typed float keeps q/k
  // in bfloat16, but a C++ array(float) is strongly float32 and would promote
  // q, k and everything downstream to float32 (profiled: +500 AsType nodes,
  // ~2.5 ms per decode step).
  return {multiply(fast::rms_norm(queries, std::nullopt, rms_epsilon),
                   array(inverse_scale * inverse_scale, queries.dtype())),
          multiply(fast::rms_norm(keys, std::nullopt, rms_epsilon),
                   array(inverse_scale, keys.dtype()))};
}

// Depthwise causal conv over the last `kernel` positions, then SiLU.
std::pair<array, array> causal_conv_step(const array& projected_qkv, const array& conv_window,
                                         const array& conv_weight, int conv_dim) {
  const array conv_input = concatenate({conv_window, projected_qkv}, 1);
  const int window_length = conv_window.shape(1);
  const array next_conv_window =
      slice(conv_input, {0, 1, 0}, {1, window_length + 1, conv_dim});
  const array conv_output = silu(conv1d(conv_input, conv_weight, 1, 0, 1, conv_dim));
  return {conv_output, next_conv_window};
}

// g = exp(-exp(A_log) · softplus(a + dt_bias)): per-head decay of the
// recurrent state, computed in float32 as the reference does.
array compute_decay_gate(const array& decay_logits, const GatedDeltaNetWeights& delta_net) {
  const array decay_rate = exp(astype(delta_net.log_decay_rate, float32));
  const array shifted_logits = astype(add(decay_logits, delta_net.decay_time_bias), float32);
  const array softplus_logits = logaddexp(shifted_logits, array(0.0f));
  return exp(negative(multiply(decay_rate, softplus_logits)));
}

// RMSNormGated: rms_norm(y) · silu(z), with the product in float32
// (mlx-lm's precise_swiglu), cast back to y's dtype.
array gated_output_norm(const array& recurrence_output, const array& output_gate,
                        const array& norm_weight, float epsilon) {
  const array normalized = fast::rms_norm(recurrence_output, norm_weight, epsilon);
  const array gated = multiply(silu(astype(output_gate, float32)),
                               astype(normalized, float32));
  return astype(gated, recurrence_output.dtype());
}

struct DeltaNetResult {
  array layer_output;
  array next_conv_window;
  array next_delta_state;
};

DeltaNetResult gated_delta_net_step(const array& normed_input,
                                    const GatedDeltaNetWeights& delta_net,
                                    const Qwen35TextConfig& config, const LayerCache& cache) {
  const int key_heads = config.delta_key_head_count;
  const int value_heads = config.delta_value_head_count;
  const int key_dim = config.delta_key_head_dim;
  const int value_dim = config.delta_value_head_dim;
  const array projected_qkv = apply_quantized_linear(normed_input, delta_net.qkv_projection);
  const array output_gate = reshape(
      apply_quantized_linear(normed_input, delta_net.gate_projection),
      {1, 1, value_heads, value_dim});
  const array beta = sigmoid(apply_quantized_linear(normed_input, delta_net.beta_projection));
  const array decay_gate = compute_decay_gate(
      apply_quantized_linear(normed_input, delta_net.decay_projection), delta_net);

  const auto [conv_output, next_conv_window] = causal_conv_step(
      projected_qkv, *cache.conv_window, delta_net.conv_weight, delta_conv_dim(config));
  const int key_width = key_heads * key_dim;
  const auto conv_parts = split(conv_output, Shape{key_width, 2 * key_width}, -1);
  const auto [queries, keys] =
      normalize_query_key(reshape(conv_parts[0], {1, 1, key_heads, key_dim}),
                          reshape(conv_parts[1], {1, 1, key_heads, key_dim}), key_dim);
  const array values = reshape(conv_parts[2], {1, 1, value_heads, value_dim});

  const auto recurrence = fast::gated_delta_update(queries, keys, values, decay_gate, beta,
                                                   *cache.delta_state);
  const array gated = gated_output_norm(recurrence[0], output_gate,
                                        delta_net.output_norm_weight, config.rms_norm_epsilon);
  const array layer_output = apply_quantized_linear(
      reshape(gated, {1, 1, value_heads * value_dim}), delta_net.output_projection);
  return {layer_output, next_conv_window, recurrence[1]};
}

struct AttentionResult {
  array layer_output;
  array next_keys;
  array next_values;
};

// Normalize per head, move heads ahead of sequence, and rotate the leading
// rotary_dims of each head at `position`.
array prepare_heads(const array& projected, const array& norm_weight, int head_count,
                    const Qwen35TextConfig& config, const array& position,
                    bool should_rotate) {
  const array per_head = reshape(projected, {1, 1, head_count, config.attention_head_dim});
  const array normed = transpose(
      fast::rms_norm(per_head, norm_weight, config.rms_norm_epsilon), {0, 2, 1, 3});
  if (!should_rotate) {
    return normed;
  }
  return fast::rope(normed, config.rotary_dims, /*traditional=*/false, config.rope_theta,
                    /*scale=*/1.0f, position);
}

AttentionResult full_attention_step(const array& normed_input,
                                    const FullAttentionWeights& attention,
                                    const Qwen35TextConfig& config, const LayerCache& cache,
                                    const array& position) {
  const int head_count = config.attention_head_count;
  const int key_value_heads = config.key_value_head_count;
  const int head_dim = config.attention_head_dim;
  // q_proj packs, per head, the query then its output gate.
  const auto query_and_gate = split(
      reshape(apply_quantized_linear(normed_input, attention.query_projection),
              {1, 1, head_count, 2 * head_dim}),
      2, -1);
  const array queries = prepare_heads(query_and_gate[0], attention.query_norm_weight,
                                      head_count, config, position, true);
  const array new_keys = prepare_heads(
      apply_quantized_linear(normed_input, attention.key_projection),
      attention.key_norm_weight, key_value_heads, config, position, true);
  const array new_values = transpose(
      reshape(apply_quantized_linear(normed_input, attention.value_projection),
              {1, 1, key_value_heads, head_dim}),
      {0, 2, 1, 3});

  const array all_keys = concatenate({*cache.cached_keys, new_keys}, 2);
  const array all_values = concatenate({*cache.cached_values, new_values}, 2);
  const float attention_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const array attended =
      fast::scaled_dot_product_attention(queries, all_keys, all_values, attention_scale);
  const array merged_heads =
      reshape(transpose(attended, {0, 2, 1, 3}), {1, 1, head_count * head_dim});
  const array output_gate = reshape(query_and_gate[1], {1, 1, head_count * head_dim});
  const array layer_output = apply_quantized_linear(
      multiply(merged_heads, sigmoid(output_gate)), attention.output_projection);
  return {layer_output, all_keys, all_values};
}

array mlp_step(const array& normed_input, const DecoderLayerWeights& layer) {
  const array gate = apply_quantized_linear(normed_input, layer.mlp_gate);
  const array up = apply_quantized_linear(normed_input, layer.mlp_up);
  return apply_quantized_linear(multiply(silu(gate), up), layer.mlp_down);
}

// Pre-norm residual block: x + mixer(norm(x)), then h + mlp(norm(h)).
array decoder_layer_step(const array& hidden, const DecoderLayerWeights& layer,
                         const Qwen35TextConfig& config, const LayerCache& cache,
                         const array& position, LayerCache& next_cache) {
  const array normed = fast::rms_norm(hidden, layer.input_norm_weight, config.rms_norm_epsilon);
  array mixer_output = normed;
  if (layer.layer_kind == LayerKind::gated_delta_net) {
    DeltaNetResult delta_result = gated_delta_net_step(normed, *layer.delta_net, config, cache);
    mixer_output = delta_result.layer_output;
    next_cache.conv_window = delta_result.next_conv_window;
    next_cache.delta_state = delta_result.next_delta_state;
  } else {
    AttentionResult attention_result =
        full_attention_step(normed, *layer.attention, config, cache, position);
    mixer_output = attention_result.layer_output;
    next_cache.cached_keys = attention_result.next_keys;
    next_cache.cached_values = attention_result.next_values;
  }
  const array residual = add(hidden, mixer_output);
  const array post_normed =
      fast::rms_norm(residual, layer.post_attention_norm_weight, config.rms_norm_epsilon);
  return add(residual, mlp_step(post_normed, layer));
}

LayerCache make_layer_cache(const Qwen35TextConfig& config, LayerKind layer_kind,
                            int context_length, bool is_random) {
  LayerCache layer_cache;
  if (layer_kind == LayerKind::gated_delta_net) {
    const Shape window_shape{1, config.delta_conv_kernel - 1, delta_conv_dim(config)};
    const Shape state_shape{1, config.delta_value_head_count, config.delta_value_head_dim,
                            config.delta_key_head_dim};
    layer_cache.conv_window = is_random ? random::normal(window_shape, ACTIVATION_DTYPE)
                                        : zeros(window_shape, ACTIVATION_DTYPE);
    // Scaled down: a unit-variance random state would saturate the recurrence.
    layer_cache.delta_state = is_random
        ? multiply(random::normal(state_shape, float32), array(0.01f))
        : zeros(state_shape, float32);
  } else {
    const Shape cache_shape{1, config.key_value_head_count, context_length,
                            config.attention_head_dim};
    layer_cache.cached_keys = is_random ? random::normal(cache_shape, ACTIVATION_DTYPE)
                                        : zeros(cache_shape, ACTIVATION_DTYPE);
    layer_cache.cached_values = is_random ? random::normal(cache_shape, ACTIVATION_DTYPE)
                                          : zeros(cache_shape, ACTIVATION_DTYPE);
  }
  return layer_cache;
}

DecodeState make_decode_state(const Qwen35TextConfig& config, int context_length,
                              bool is_random) {
  DecodeState decode_state{{}, array(context_length, int32)};
  for (const LayerKind layer_kind : config.layer_kinds) {
    decode_state.layer_caches.push_back(
        make_layer_cache(config, layer_kind, context_length, is_random));
  }
  return decode_state;
}

// The two arrays a layer's cache contributes, in flattening order.
std::pair<array, array> layer_cache_pair(const LayerCache& layer_cache, LayerKind layer_kind) {
  return layer_kind == LayerKind::gated_delta_net
      ? std::pair<array, array>{*layer_cache.conv_window, *layer_cache.delta_state}
      : std::pair<array, array>{*layer_cache.cached_keys, *layer_cache.cached_values};
}

std::vector<LayerKind> layer_kinds_of(const DecodeState& decode_state) {
  std::vector<LayerKind> layer_kinds;
  for (const LayerCache& layer_cache : decode_state.layer_caches) {
    layer_kinds.push_back(layer_cache.conv_window ? LayerKind::gated_delta_net
                                                  : LayerKind::full_attention);
  }
  return layer_kinds;
}

void append_state_arrays(const DecodeState& decode_state, std::vector<array>& flat_arrays) {
  const std::vector<LayerKind> layer_kinds = layer_kinds_of(decode_state);
  for (std::size_t layer_index = 0; layer_index < layer_kinds.size(); ++layer_index) {
    const auto [first_array, second_array] =
        layer_cache_pair(decode_state.layer_caches[layer_index], layer_kinds[layer_index]);
    flat_arrays.push_back(first_array);
    flat_arrays.push_back(second_array);
  }
}

} // namespace

DecodeState make_empty_decode_state(const Qwen35TextConfig& config) {
  return make_decode_state(config, 0, false);
}

DecodeState make_synthetic_decode_state(const Qwen35TextConfig& config, int context_length) {
  if (context_length < 0) {
    throw std::invalid_argument("context_length must be non-negative");
  }
  return make_decode_state(config, context_length, true);
}

DecodeStepResult run_decode_step(const Qwen35Weights& weights, const Qwen35TextConfig& config,
                                 const array& token_ids, const DecodeState& decode_state) {
  if (decode_state.layer_caches.size() != weights.layers.size()) {
    throw std::invalid_argument("decode state and weights disagree on layer count");
  }
  array hidden = reshape(embed_tokens(weights.token_embedding, token_ids),
                         {1, 1, config.hidden_size});
  DecodeState next_state{std::vector<LayerCache>(weights.layers.size()),
                         add(decode_state.position, array(1, int32))};
  for (std::size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
    hidden = decoder_layer_step(hidden, weights.layers[layer_index], config,
                                decode_state.layer_caches[layer_index], decode_state.position,
                                next_state.layer_caches[layer_index]);
  }
  const array final_hidden =
      fast::rms_norm(hidden, weights.final_norm_weight, config.rms_norm_epsilon);
  // Tied embeddings: the LM head is the embedding table, transposed.
  return {apply_quantized_linear(final_hidden, weights.token_embedding), next_state};
}

std::vector<array> flatten_step_inputs(const array& token_ids, const DecodeState& decode_state) {
  std::vector<array> flat_inputs{token_ids, decode_state.position};
  append_state_arrays(decode_state, flat_inputs);
  return flat_inputs;
}

std::vector<array> flatten_step_outputs(const DecodeStepResult& step_result) {
  std::vector<array> flat_outputs{step_result.logits, step_result.next_state.position};
  append_state_arrays(step_result.next_state, flat_outputs);
  return flat_outputs;
}

DecodeState unflatten_state(const std::vector<array>& state_arrays, std::size_t first_index,
                            const Qwen35TextConfig& config) {
  const std::size_t expected_size =
      first_index + 1 + ARRAYS_PER_LAYER_CACHE * config.layer_kinds.size();
  if (state_arrays.size() != expected_size) {
    throw std::invalid_argument("flattened decode state has the wrong length");
  }
  DecodeState decode_state{{}, state_arrays[first_index]};
  std::size_t array_index = first_index + 1;
  for (const LayerKind layer_kind : config.layer_kinds) {
    LayerCache layer_cache;
    if (layer_kind == LayerKind::gated_delta_net) {
      layer_cache.conv_window = state_arrays[array_index];
      layer_cache.delta_state = state_arrays[array_index + 1];
    } else {
      layer_cache.cached_keys = state_arrays[array_index];
      layer_cache.cached_values = state_arrays[array_index + 1];
    }
    decode_state.layer_caches.push_back(layer_cache);
    array_index += ARRAYS_PER_LAYER_CACHE;
  }
  return decode_state;
}

DecodeStepResult unflatten_step_outputs(const std::vector<array>& step_outputs,
                                        const Qwen35TextConfig& config) {
  if (step_outputs.empty()) {
    throw std::invalid_argument("decode step produced no outputs");
  }
  return {step_outputs[0], unflatten_state(step_outputs, 1, config)};
}

} // namespace mlx_plan::qwen35
