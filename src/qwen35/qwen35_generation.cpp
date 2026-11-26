#include "qwen35_generation.h"

#include "mlx/ops.h"
#include "mlx/transforms.h"

#include <stdexcept>

namespace mlx_perf::qwen35 {
namespace {

using namespace mlx::core;

// Evaluates the logits and every next-state array together, so the state a
// later step reads is materialized and the graph never grows across steps.
DecodeStepResult step_and_materialize(const Qwen35Weights& weights,
                                      const Qwen35TextConfig& config, int token_id,
                                      const DecodeState& decode_state) {
  DecodeStepResult step_result =
      run_decode_step(weights, config, array({token_id}, {1, 1}), decode_state);
  eval(flatten_step_outputs(step_result));
  return step_result;
}

int argmax_token(const array& logits) {
  return argmax(reshape(logits, {-1})).item<uint32_t>();
}

} // namespace

std::vector<int> generate_greedy(const Qwen35Weights& weights, const Qwen35TextConfig& config,
                                 const std::vector<int>& prompt_token_ids, int new_token_count) {
  if (prompt_token_ids.empty() || new_token_count < 0) {
    throw std::invalid_argument("generation needs a prompt and a non-negative token count");
  }
  DecodeState decode_state = make_empty_decode_state(config);
  DecodeStepResult step_result{array(0.0f), decode_state};
  for (const int prompt_token : prompt_token_ids) {
    step_result = step_and_materialize(weights, config, prompt_token, decode_state);
    decode_state = step_result.next_state;
  }
  std::vector<int> generated_token_ids;
  for (int generated_count = 0; generated_count < new_token_count; ++generated_count) {
    const int next_token = argmax_token(step_result.logits);
    generated_token_ids.push_back(next_token);
    step_result = step_and_materialize(weights, config, next_token, decode_state);
    decode_state = step_result.next_state;
  }
  return generated_token_ids;
}

} // namespace mlx_perf::qwen35
