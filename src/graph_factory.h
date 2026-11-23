#pragma once

// Graph factory: the seam between "what job to measure" and "how to measure
// it". Every measurement pass needs its own freshly built, unevaluated graph
// (evaluation materializes and detaches nodes), so a pass asks a factory.

#include "mlx/array.h"

#include <string>
#include <vector>

namespace mlx_plan {

// A cut through the job, in Spark's sense of a stage boundary: everything
// the boundary arrays depend on that an EARLIER boundary did not already
// claim forms one stage. The profiler stays model-agnostic — only the job
// knows where its natural cuts are (for a decoder, the end of each layer).
struct StageBoundary {
  std::string stage_name;
  std::vector<mlx::core::array> boundary_arrays;
};

// A fresh graph plus the job's own stage cuts, in execution order. Nodes no
// boundary reaches form a final stage of their own.
struct StagedGraph {
  std::vector<mlx::core::array> output_arrays;
  std::vector<StageBoundary> stage_boundaries;
  // Names the stage of nodes no boundary reaches (e.g. "LM head").
  std::string final_stage_name = "rest of job";
};

class GraphFactory {
 public:
  virtual ~GraphFactory() = default;
  // Returns the job's output arrays, unevaluated, on a brand-new graph.
  virtual std::vector<mlx::core::array> build_fresh_graph() const = 0;
  // The same graph with stage cuts. Jobs with no natural stages need not
  // override it: no cuts means the profiler times nodes individually unless
  // asked otherwise.
  virtual StagedGraph build_fresh_staged_graph() const {
    return {build_fresh_graph(), {}, "whole job"};
  }
};

} // namespace mlx_plan
