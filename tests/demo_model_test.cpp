// Integration tests for demo_model on the Metal GPU: the eager and compiled
// variants must compute the same thing (compile is observed, never allowed
// to change semantics) while compile shrinks the plan.

#include "demo_model.h"
#include "graph_model.h"
#include "test_harness.h"

#include "mlx/mlx.h"

#include <set>
#include <stdexcept>
#include <string>

namespace {

using namespace mlx_plan;

struct DemoFixture {
  DemoStreams demo_streams = make_demo_streams();
  std::vector<mlx::core::array> demo_inputs =
      make_materialized_demo_inputs(demo_streams);
};

std::set<int> streams_used(const CapturedGraph& captured_graph) {
  std::set<int> stream_indices;
  for (const std::size_t node_index : captured_graph.topological) {
    stream_indices.insert(captured_graph.nodes[node_index].stream_index);
  }
  return stream_indices;
}

// --- positive ---

void should_compute_identical_results_eagerly_and_compiled() {
  DemoFixture fixture;
  const EagerDemoGraphFactory eager_factory(fixture.demo_streams, fixture.demo_inputs);
  const CompiledDemoGraphFactory compiled_factory(fixture.demo_streams,
                                                  fixture.demo_inputs);
  const auto eager_outputs = eager_factory.build_fresh_graph();
  const auto compiled_outputs = compiled_factory.build_fresh_graph();
  EXPECT_TRUE(eager_outputs.size() == 1 && compiled_outputs.size() == 1);
  EXPECT_TRUE(mlx::core::allclose(eager_outputs[0], compiled_outputs[0], 1e-4, 1e-6)
                  .item<bool>());
}

void should_fuse_into_fewer_primitives_on_the_same_streams() {
  DemoFixture fixture;
  const CapturedGraph eager_graph = capture_graph(
      EagerDemoGraphFactory(fixture.demo_streams, fixture.demo_inputs)
          .build_fresh_graph());
  const CapturedGraph compiled_graph = capture_graph(
      CompiledDemoGraphFactory(fixture.demo_streams, fixture.demo_inputs)
          .build_fresh_graph());
  EXPECT_TRUE(compiled_graph.topological.size() < eager_graph.topological.size());
  EXPECT_TRUE(streams_used(compiled_graph) == streams_used(eager_graph));
  bool has_fused_node = false;
  for (const std::size_t node_index : compiled_graph.topological) {
    has_fused_node |=
        compiled_graph.nodes[node_index].op_name.rfind("Compiled", 0) == 0;
  }
  EXPECT_TRUE(has_fused_node);
}

// --- edge / boundary ---

void should_build_a_brand_new_unevaluated_graph_on_every_call() {
  DemoFixture fixture;
  const EagerDemoGraphFactory eager_factory(fixture.demo_streams, fixture.demo_inputs);
  const auto first_outputs = eager_factory.build_fresh_graph();
  const auto second_outputs = eager_factory.build_fresh_graph();
  EXPECT_TRUE(first_outputs[0].id() != second_outputs[0].id());
  EXPECT_TRUE(first_outputs[0].has_primitive());
  EXPECT_TRUE(!first_outputs[0].is_available());
}

void should_put_main_stream_first_and_branches_on_three_streams() {
  DemoFixture fixture;
  EXPECT_TRUE(fixture.demo_streams.stream_index_to_stream.size() == 3);
  EXPECT_TRUE(fixture.demo_streams.main_stream.index <
              fixture.demo_streams.first_branch_stream.index);
  EXPECT_TRUE(fixture.demo_inputs.size() == 4);
  for (const mlx::core::array& demo_input : fixture.demo_inputs) {
    EXPECT_TRUE(demo_input.is_available());
  }
}

// --- negative ---

void should_reject_a_wrong_number_of_inputs() {
  DemoFixture fixture;
  const DemoComputation demo_computation(fixture.demo_streams);
  bool has_thrown = false;
  try {
    demo_computation({fixture.demo_inputs[0]});
  } catch (const std::invalid_argument&) {
    has_thrown = true;
  }
  EXPECT_TRUE(has_thrown);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_compute_identical_results_eagerly_and_compiled),
      TEST_CASE(should_fuse_into_fewer_primitives_on_the_same_streams),
      TEST_CASE(should_build_a_brand_new_unevaluated_graph_on_every_call),
      TEST_CASE(should_put_main_stream_first_and_branches_on_three_streams),
      TEST_CASE(should_reject_a_wrong_number_of_inputs),
  });
}
