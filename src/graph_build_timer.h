#pragma once

// Graph build timer: one concern only — time how long the job's own code
// takes to build each fresh graph, per measurement pass. Graph construction
// is host time a profiler of a short job should show, and it happens between
// the passes' clocks, so no pass measures it otherwise.
//
// A GraphFactory decorator rather than timing inside the passes: the passes
// stay unaware of it, and the same decorator wraps C++ factories and the C
// API's callback alike.

#include "graph_factory.h"

#include <string>
#include <vector>

namespace mlx_perf {

// Each call's duration, in microseconds, for one pass, in call order.
struct GraphBuildPass {
  std::string pass_name;
  std::vector<double> call_microseconds;
};

class TimedGraphFactory final : public GraphFactory {
 public:
  // `timed_factory` must outlive this decorator.
  explicit TimedGraphFactory(const GraphFactory& timed_factory);

  // Files every later call under `pass_name` until the next begin_pass.
  // Calls before the first begin_pass are filed under "unlabelled".
  void begin_pass(const std::string& pass_name);

  std::vector<mlx::core::array> build_fresh_graph() const override;
  StagedGraph build_fresh_staged_graph() const override;

  // Passes in the order they began; a pass with no calls is kept, empty.
  const std::vector<GraphBuildPass>& build_passes() const { return build_passes_; }

 private:
  void record_call(double call_microseconds) const;

  const GraphFactory& timed_factory_;
  // Mutable: GraphFactory's build methods are const (building a graph does
  // not change the job), but timing them is this decorator's whole purpose.
  mutable std::vector<GraphBuildPass> build_passes_;
};

} // namespace mlx_perf
