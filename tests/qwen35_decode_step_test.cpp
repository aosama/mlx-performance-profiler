// Integration tests (Metal GPU) for the Qwen3.5 decode step, on a tiny
// synthetic model: random weights quantized by MLX itself, one Gated
// DeltaNet layer and one full-attention layer. No checkpoint files needed.

#include "qwen35/qwen35_decode_job.h"
#include "qwen35/qwen35_decode_step.h"
#include "test_harness.h"
#include "tiny_qwen35_model.h"

#include "mlx/compile.h"
#include "mlx/mlx.h"

#include <stdexcept>

namespace {

using namespace mlx::core;
using namespace mlx_perf::qwen35;

using namespace tiny_qwen35_model;

array token(int token_id) { return array({token_id}, {1, 1}); }

// MLX's const item() refuses unevaluated arrays; evaluate before reading.
template <typename Scalar>
Scalar read_scalar(array scalar_array) {
  scalar_array.eval();
  return scalar_array.item<Scalar>();
}

// --- positive ---

void should_produce_vocabulary_logits_and_advance_every_cache() {
  const Qwen35TextConfig config = make_tiny_config();
  const Qwen35Weights weights = make_tiny_weights(config);
  const DecodeState state = make_synthetic_decode_state(config, 7);
  const DecodeStepResult step_result = run_decode_step(weights, config, token(3), state);
  EXPECT_TRUE(step_result.logits.shape() == Shape({1, 1, TINY_VOCABULARY}));
  EXPECT_TRUE(read_scalar<bool>(all(isfinite(step_result.logits))));
  EXPECT_TRUE(read_scalar<int>(step_result.next_state.position) == 8);
  EXPECT_TRUE(step_result.next_state.layer_caches[1].cached_keys->shape(2) == 8);
  EXPECT_TRUE(step_result.next_state.layer_caches[0].delta_state->shape() ==
              state.layer_caches[0].delta_state->shape());
}

void should_match_eager_logits_when_compiled() {
  const Qwen35TextConfig config = make_tiny_config();
  const Qwen35Weights weights = make_tiny_weights(config);
  const auto flat_inputs = make_profiling_inputs(config, 5, 9);
  const DecodeStepFunction decode_step(weights, config);
  const auto compiled_step = compile(DecodeStepFunction(weights, config));
  const array eager_logits = decode_step(flat_inputs)[0];
  const array compiled_logits = compiled_step(flat_inputs)[0];
  EXPECT_TRUE(read_scalar<bool>(
      allclose(astype(eager_logits, float32), astype(compiled_logits, float32), 1e-2, 1e-2)));
}

// --- edge / boundary ---

void should_decode_from_an_empty_state_at_position_zero() {
  const Qwen35TextConfig config = make_tiny_config();
  const Qwen35Weights weights = make_tiny_weights(config);
  const DecodeStepResult step_result =
      run_decode_step(weights, config, token(0), make_empty_decode_state(config));
  EXPECT_TRUE(read_scalar<int>(step_result.next_state.position) == 1);
  EXPECT_TRUE(step_result.next_state.layer_caches[1].cached_keys->shape(2) == 1);
}

void should_round_trip_state_through_flattening() {
  const Qwen35TextConfig config = make_tiny_config();
  const DecodeState state = make_synthetic_decode_state(config, 3);
  const auto flat_inputs = flatten_step_inputs(token(1), state);
  EXPECT_TRUE(flat_inputs.size() == 2 + 2 * config.layer_kinds.size());
  const DecodeState restored = unflatten_state(flat_inputs, 1, config);
  EXPECT_TRUE(restored.layer_caches[0].conv_window->id() == state.layer_caches[0].conv_window->id());
  EXPECT_TRUE(restored.layer_caches[1].cached_values->id() ==
              state.layer_caches[1].cached_values->id());
}

void should_derive_bit_width_from_packed_shapes() {
  EXPECT_TRUE(derive_quantization_bits(make_linear(8, 128, 4).packed_weight,
                                       make_linear(8, 128, 4).scales, 64) == 4);
  const QuantizedLinear eight_bit = make_linear(8, 128, 8);
  EXPECT_TRUE(derive_quantization_bits(eight_bit.packed_weight, eight_bit.scales, 64) == 8);
}

// --- negative ---

void should_reject_state_and_weights_that_disagree() {
  const Qwen35TextConfig config = make_tiny_config();
  const Qwen35Weights weights = make_tiny_weights(config);
  Qwen35TextConfig one_layer_config = config;
  one_layer_config.layer_kinds = {LayerKind::gated_delta_net};
  bool has_thrown = false;
  try {
    run_decode_step(weights, config, token(0), make_empty_decode_state(one_layer_config));
  } catch (const std::invalid_argument&) {
    has_thrown = true;
  }
  EXPECT_TRUE(has_thrown);
}

void should_reject_malformed_flattened_state_and_shapes() {
  const Qwen35TextConfig config = make_tiny_config();
  int thrown_count = 0;
  try {
    unflatten_state({token(0), array(0, int32)}, 1, config);
  } catch (const std::invalid_argument&) {
    ++thrown_count;
  }
  try {
    make_synthetic_decode_state(config, -1);
  } catch (const std::invalid_argument&) {
    ++thrown_count;
  }
  try {
    derive_quantization_bits(zeros({8}), zeros({8}), 64);
  } catch (const std::runtime_error&) {
    ++thrown_count;
  }
  EXPECT_TRUE(thrown_count == 3);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_produce_vocabulary_logits_and_advance_every_cache),
      TEST_CASE(should_match_eager_logits_when_compiled),
      TEST_CASE(should_decode_from_an_empty_state_at_position_zero),
      TEST_CASE(should_round_trip_state_through_flattening),
      TEST_CASE(should_derive_bit_width_from_packed_shapes),
      TEST_CASE(should_reject_state_and_weights_that_disagree),
      TEST_CASE(should_reject_malformed_flattened_state_and_shapes),
  });
}
