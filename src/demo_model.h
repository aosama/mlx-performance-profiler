#pragma once

// Demo model: one concern only — the toy "job" the demo measures: a pure
// three-stream computation over pre-materialized inputs, buildable eagerly or
// through mlx::core::compile, so both variants run the identical math.
//
// Why inputs are materialized up front: generating them inside the graph
// (random::normal) made ~70 of 81 nodes RNG plumbing, drowning the signal,
// and compile needs a pure function of its inputs anyway. Spark analog: the
// job reads a cached dataset instead of regenerating it every stage.

#include "graph_factory.h"

#include "mlx/array.h"
#include "mlx/stream.h"

#include <functional>
#include <unordered_map>
#include <vector>

namespace mlx_plan {

struct DemoStreams {
  mlx::core::Stream main_stream;
  mlx::core::Stream first_branch_stream;
  mlx::core::Stream second_branch_stream;
  std::unordered_map<int, mlx::core::Stream> stream_index_to_stream;
};

// Touches the main stream FIRST: MLX numbers streams in creation order and
// creates the default stream lazily, and the report assumes main = 0.
DemoStreams make_demo_streams();

// Random inputs on the main stream, evaluated before any measurement.
std::vector<mlx::core::array> make_materialized_demo_inputs(
    const DemoStreams& demo_streams);

// The job itself. A callable object (not a lambda) so it can be handed to
// mlx::core::compile while its stream bindings stay explicit and named.
class DemoComputation {
 public:
  explicit DemoComputation(const DemoStreams& demo_streams);
  std::vector<mlx::core::array> operator()(
      const std::vector<mlx::core::array>& demo_inputs) const;

 private:
  mlx::core::Stream main_stream_;
  mlx::core::Stream first_branch_stream_;
  mlx::core::Stream second_branch_stream_;
};

class EagerDemoGraphFactory final : public GraphFactory {
 public:
  EagerDemoGraphFactory(
      const DemoStreams& demo_streams,
      std::vector<mlx::core::array> demo_inputs);
  std::vector<mlx::core::array> build_fresh_graph() const override;

 private:
  DemoComputation demo_computation_;
  std::vector<mlx::core::array> demo_inputs_;
};

// Each call of the compiled function returns a fresh graph whose fusable
// same-stream chains are single `Compiled…` primitives; the first call
// traces and is absorbed by the warm-up pass.
class CompiledDemoGraphFactory final : public GraphFactory {
 public:
  CompiledDemoGraphFactory(
      const DemoStreams& demo_streams,
      std::vector<mlx::core::array> demo_inputs);
  std::vector<mlx::core::array> build_fresh_graph() const override;

 private:
  std::function<std::vector<mlx::core::array>(
      const std::vector<mlx::core::array>&)>
      compiled_computation_;
  std::vector<mlx::core::array> demo_inputs_;
};

} // namespace mlx_plan
