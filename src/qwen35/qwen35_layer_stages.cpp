#include "qwen35_layer_stages.h"

#include <stdexcept>
#include <utility>

namespace mlx_perf::qwen35 {
namespace {

constexpr std::size_t LEADING_OUTPUT_COUNT = 2;  // logits, position
constexpr std::size_t STATE_ARRAYS_PER_LAYER = 2;

// Layer 0's stage also claims the token embedding, its only producer.
std::string eager_stage_name(std::size_t layer_index) {
  return layer_index == 0 ? "embed + layer 0" : "layer " + std::to_string(layer_index);
}

std::vector<mlx::core::array> present_cache_arrays(const LayerCache& layer_cache) {
  std::vector<mlx::core::array> cache_arrays;
  for (const auto* cache_entry : {&layer_cache.conv_window, &layer_cache.delta_state,
                                  &layer_cache.cached_keys, &layer_cache.cached_values}) {
    if (cache_entry->has_value()) {
      cache_arrays.push_back(**cache_entry);
    }
  }
  return cache_arrays;
}

} // namespace

void LayerStageCollector::on_layer_built(std::size_t layer_index,
                                         const mlx::core::array& layer_output,
                                         const LayerCache& next_layer_cache) {
  StageBoundary stage_boundary{eager_stage_name(layer_index), {layer_output}};
  for (const auto& cache_array : present_cache_arrays(next_layer_cache)) {
    stage_boundary.boundary_arrays.push_back(cache_array);
  }
  stage_boundaries_.push_back(std::move(stage_boundary));
}

std::vector<StageBoundary> LayerStageCollector::take_stage_boundaries() {
  return std::exchange(stage_boundaries_, {});
}

std::vector<StageBoundary> layer_state_boundaries(
    const std::vector<mlx::core::array>& flat_step_outputs, const Qwen35TextConfig& config) {
  const std::size_t layer_count = config.layer_kinds.size();
  if (flat_step_outputs.size() != LEADING_OUTPUT_COUNT + STATE_ARRAYS_PER_LAYER * layer_count) {
    throw std::invalid_argument("compiled step outputs do not match the layer count");
  }
  std::vector<StageBoundary> stage_boundaries;
  for (std::size_t layer_index = 0; layer_index < layer_count; ++layer_index) {
    const std::size_t first_state_index =
        LEADING_OUTPUT_COUNT + STATE_ARRAYS_PER_LAYER * layer_index;
    stage_boundaries.push_back(
        {"up to layer " + std::to_string(layer_index) + " state",
         {flat_step_outputs[first_state_index], flat_step_outputs[first_state_index + 1]}});
  }
  return stage_boundaries;
}

} // namespace mlx_perf::qwen35
