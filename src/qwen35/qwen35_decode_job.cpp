#include "qwen35_decode_job.h"

#include "qwen35_layer_stages.h"

#include "mlx/compile.h"
#include "mlx/transforms.h"

#include <utility>

namespace mlx_plan::qwen35 {

using mlx::core::array;

std::vector<array> make_profiling_inputs(const Qwen35TextConfig& config, int context_length,
                                         int token_id) {
  std::vector<array> flat_inputs = flatten_step_inputs(
      array({token_id}, {1, 1}), make_synthetic_decode_state(config, context_length));
  // Materialized now: generating the random state must not be measured.
  mlx::core::eval(flat_inputs);
  return flat_inputs;
}

DecodeStepFunction::DecodeStepFunction(const Qwen35Weights& weights,
                                       const Qwen35TextConfig& config)
    : weights_(&weights), config_(&config) {}

std::vector<array> DecodeStepFunction::operator()(const std::vector<array>& flat_inputs) const {
  const DecodeState decode_state = unflatten_state(flat_inputs, 1, *config_);
  return flatten_step_outputs(
      run_decode_step(*weights_, *config_, flat_inputs.front(), decode_state));
}

std::vector<array> DecodeStepFunction::run_observed(const std::vector<array>& flat_inputs,
                                                    DecoderLayerObserver& layer_observer) const {
  const DecodeState decode_state = unflatten_state(flat_inputs, 1, *config_);
  return flatten_step_outputs(run_decode_step(*weights_, *config_, flat_inputs.front(),
                                              decode_state, &layer_observer));
}

EagerDecodeGraphFactory::EagerDecodeGraphFactory(const Qwen35Weights& weights,
                                                 const Qwen35TextConfig& config,
                                                 std::vector<array> flat_inputs)
    : decode_step_(weights, config), flat_inputs_(std::move(flat_inputs)) {}

std::vector<array> EagerDecodeGraphFactory::build_fresh_graph() const {
  return decode_step_(flat_inputs_);
}

StagedGraph EagerDecodeGraphFactory::build_fresh_staged_graph() const {
  LayerStageCollector layer_stage_collector;
  std::vector<array> step_outputs = decode_step_.run_observed(flat_inputs_, layer_stage_collector);
  return {std::move(step_outputs), layer_stage_collector.take_stage_boundaries(),
          FINAL_STAGE_NAME};
}

// Weights are not compile inputs, so the trace captures them as constants —
// exactly how mlx-lm compiles its step functions.
CompiledDecodeGraphFactory::CompiledDecodeGraphFactory(const Qwen35Weights& weights,
                                                       const Qwen35TextConfig& config,
                                                       std::vector<array> flat_inputs)
    : config_(&config),
      compiled_decode_step_(mlx::core::compile(DecodeStepFunction(weights, config))),
      flat_inputs_(std::move(flat_inputs)) {}

std::vector<array> CompiledDecodeGraphFactory::build_fresh_graph() const {
  return compiled_decode_step_(flat_inputs_);
}

StagedGraph CompiledDecodeGraphFactory::build_fresh_staged_graph() const {
  std::vector<array> step_outputs = compiled_decode_step_(flat_inputs_);
  std::vector<StageBoundary> stage_boundaries = layer_state_boundaries(step_outputs, *config_);
  return {std::move(step_outputs), std::move(stage_boundaries), FINAL_STAGE_NAME};
}

} // namespace mlx_plan::qwen35
