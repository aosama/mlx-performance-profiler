#pragma once

// Qwen3.5 decode step: one concern only — the forward math for ONE token
// through the whole dense model (embedding → Gated DeltaNet / gated full
// attention layers → final norm → tied LM head), matching mlx-lm's
// qwen3_5.py, built only from MLX's own production kernels
// (quantized_matmul, fast::rms_norm, fast::rope,
// fast::scaled_dot_product_attention, fast::gated_delta_update).
//
// The step is a pure function of (token, state) → (logits, next state), so
// mlx::core::compile can trace it; the rope position is an array input for
// the same reason (an int would be baked into the trace).

#include "qwen35_config.h"
#include "qwen35_weights.h"

#include "mlx/array.h"

#include <optional>
#include <vector>

namespace mlx_perf::qwen35 {

// Per-layer recurrent/cache state. Gated DeltaNet layers use the first two,
// full-attention layers the last two.
struct LayerCache {
  std::optional<mlx::core::array> conv_window;     // [1, kernel-1, conv_dim]
  std::optional<mlx::core::array> delta_state;     // [1, Hv, Dv, Dk] float32
  std::optional<mlx::core::array> cached_keys;     // [1, Hkv, past, D]
  std::optional<mlx::core::array> cached_values;   // [1, Hkv, past, D]
};

struct DecodeState {
  std::vector<LayerCache> layer_caches;
  mlx::core::array position;  // int32 scalar: tokens already consumed
};

struct DecodeStepResult {
  mlx::core::array logits;  // [1, 1, vocabulary]
  DecodeState next_state;
};

// Told each decoder layer's outputs as the step is BUILT (lazily, before any
// evaluation) — how a profiler learns where one layer ends while the model
// math stays free of profiling concerns.
class DecoderLayerObserver {
 public:
  virtual ~DecoderLayerObserver() = default;
  virtual void on_layer_built(std::size_t layer_index,
                              const mlx::core::array& layer_output,
                              const LayerCache& next_layer_cache) = 0;
};

// Zero state at position 0, for real generation.
DecodeState make_empty_decode_state(const Qwen35TextConfig& config);

// Random-valued state as if `context_length` tokens had been consumed: the
// shapes (and so the work) of a decode step deep into a conversation. Values
// are meaningless; only timing is measured with it.
DecodeState make_synthetic_decode_state(const Qwen35TextConfig& config,
                                        int context_length);

DecodeStepResult run_decode_step(const Qwen35Weights& weights,
                                 const Qwen35TextConfig& config,
                                 const mlx::core::array& token_ids,  // [1, 1] int32
                                 const DecodeState& decode_state,
                                 DecoderLayerObserver* layer_observer = nullptr);

// Flattening for mlx::core::compile, which traces vector<array> → vector<array>.
// Order: token_ids, position, then each layer's two cache arrays.
std::vector<mlx::core::array> flatten_step_inputs(const mlx::core::array& token_ids,
                                                  const DecodeState& decode_state);
std::vector<mlx::core::array> flatten_step_outputs(const DecodeStepResult& step_result);
DecodeStepResult unflatten_step_outputs(const std::vector<mlx::core::array>& step_outputs,
                                        const Qwen35TextConfig& config);
DecodeState unflatten_state(const std::vector<mlx::core::array>& state_arrays,
                            std::size_t first_index, const Qwen35TextConfig& config);

} // namespace mlx_perf::qwen35
