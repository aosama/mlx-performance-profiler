#pragma once

// Graph factory: the seam between "what job to measure" and "how to measure
// it". Every measurement pass needs its own freshly built, unevaluated graph
// (evaluation materializes and detaches nodes), so a pass asks a factory.

#include "mlx/array.h"

#include <vector>

namespace mlx_plan {

class GraphFactory {
 public:
  virtual ~GraphFactory() = default;
  // Returns the job's output arrays, unevaluated, on a brand-new graph.
  virtual std::vector<mlx::core::array> build_fresh_graph() const = 0;
};

} // namespace mlx_plan
