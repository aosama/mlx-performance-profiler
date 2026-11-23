#pragma once

// Qwen3.5 layer stages: one concern only — tell the generic profiler where
// each decoder layer ends (Spark's stage = one layer here), so the timeline
// pass can time ~45 nodes per dispatch instead of one.
//
// Eager graphs expose every layer's hidden state, so a stage is exactly one
// layer. mlx::core::compile hides intermediates: only the step's OUTPUTS
// (logits, next state) survive, so compiled stages are cut at each layer's
// next-state outputs. Work that feeds only the next hidden state (a layer's
// output projection and MLP) then lands in the FOLLOWING stage; the stage
// names say so rather than claiming a clean layer.

#include "graph_factory.h"
#include "qwen35_config.h"
#include "qwen35_decode_step.h"

#include <string>
#include <vector>

namespace mlx_plan::qwen35 {

// Collects one boundary per layer while an eager step is being built: the
// layer's output hidden state plus its next cache arrays.
class LayerStageCollector final : public DecoderLayerObserver {
 public:
  void on_layer_built(std::size_t layer_index, const mlx::core::array& layer_output,
                      const LayerCache& next_layer_cache) override;
  std::vector<StageBoundary> take_stage_boundaries();

 private:
  std::vector<StageBoundary> stage_boundaries_;
};

// Compiled-step cuts from the flat outputs (flatten_step_outputs order:
// logits, position, then two state arrays per layer).
// Throws std::invalid_argument when the outputs do not match `config`.
std::vector<StageBoundary> layer_state_boundaries(
    const std::vector<mlx::core::array>& flat_step_outputs, const Qwen35TextConfig& config);

// What runs after the last layer's cut: final norm and the tied LM head.
inline const std::string FINAL_STAGE_NAME = "final norm + LM head";

} // namespace mlx_plan::qwen35
