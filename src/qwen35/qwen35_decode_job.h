#pragma once

// Qwen3.5 decode job: one concern only — present one real decode step as
// GraphFactory implementations (eager, and through mlx::core::compile) so
// the generic measurement session can profile it like any other job.

#include "graph_factory.h"
#include "qwen35_decode_step.h"

#include <functional>
#include <vector>

namespace mlx_plan::qwen35 {

// One materialized token and decode state, as if `context_length` tokens of
// conversation had been consumed. Shared by both variants so they measure
// identical work.
std::vector<mlx::core::array> make_profiling_inputs(const Qwen35TextConfig& config,
                                                    int context_length, int token_id);

// The decode step as compile needs it: flat arrays in, flat arrays out.
// A callable object rather than a lambda so its captures are explicit.
class DecodeStepFunction {
 public:
  DecodeStepFunction(const Qwen35Weights& weights, const Qwen35TextConfig& config);
  std::vector<mlx::core::array> operator()(
      const std::vector<mlx::core::array>& flat_inputs) const;

 private:
  const Qwen35Weights* weights_;
  const Qwen35TextConfig* config_;
};

class EagerDecodeGraphFactory final : public GraphFactory {
 public:
  EagerDecodeGraphFactory(const Qwen35Weights& weights, const Qwen35TextConfig& config,
                          std::vector<mlx::core::array> flat_inputs);
  std::vector<mlx::core::array> build_fresh_graph() const override;

 private:
  DecodeStepFunction decode_step_;
  std::vector<mlx::core::array> flat_inputs_;
};

class CompiledDecodeGraphFactory final : public GraphFactory {
 public:
  CompiledDecodeGraphFactory(const Qwen35Weights& weights, const Qwen35TextConfig& config,
                             std::vector<mlx::core::array> flat_inputs);
  std::vector<mlx::core::array> build_fresh_graph() const override;

 private:
  std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)>
      compiled_decode_step_;
  std::vector<mlx::core::array> flat_inputs_;
};

} // namespace mlx_plan::qwen35
