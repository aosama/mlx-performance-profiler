// Unit tests for graph_model: the traversal every other module consumes.
// Graphs are built but never evaluated, so no GPU work runs.

#include "graph_model.h"
#include "test_harness.h"

#include "mlx/mlx.h"

#include <unordered_map>

namespace {

using namespace mlx::core;
using mlx_perf::capture_graph;
using mlx_perf::CapturedGraph;

std::unordered_map<std::uintptr_t, std::size_t> topological_rank(
    const CapturedGraph& captured_graph) {
  std::unordered_map<std::uintptr_t, std::size_t> array_id_to_rank;
  for (std::size_t rank = 0; rank < captured_graph.topological.size(); ++rank) {
    array_id_to_rank[captured_graph.nodes[captured_graph.topological[rank]].array_id] = rank;
  }
  return array_id_to_rank;
}

// --- positive ---

void should_order_every_primitive_after_its_primitive_inputs() {
  const array left_input = ones({8, 8});
  const array product = matmul(left_input, left_input);
  const array activated = exp(add(product, left_input));
  const CapturedGraph captured_graph = capture_graph({activated});
  const auto array_id_to_rank = topological_rank(captured_graph);
  for (const std::size_t node_index : captured_graph.topological) {
    const auto& graph_node = captured_graph.nodes[node_index];
    for (const auto& input : graph_node.handle.inputs()) {
      const auto input_rank = array_id_to_rank.find(input.id());
      if (input_rank != array_id_to_rank.end()) {
        EXPECT_TRUE(input_rank->second < array_id_to_rank.at(graph_node.array_id));
      }
    }
  }
  EXPECT_TRUE(captured_graph.nodes[captured_graph.topological.back()].op_name == "Exp");
}

void should_label_edges_with_bytes_read_and_collect_streams() {
  const Stream side_stream = new_stream(Device::gpu);
  const array side_output = exp(ones({4, 4}, float32, side_stream), side_stream);
  const array joined = add(side_output, side_output);
  const CapturedGraph captured_graph = capture_graph({joined});
  bool has_side_edge = false;
  for (const auto& graph_edge : captured_graph.edges) {
    has_side_edge |= graph_edge.bytes == 4 * 4 * sizeof(float);
  }
  EXPECT_TRUE(has_side_edge);
  const auto stream_index_to_stream = mlx_perf::collect_streams(captured_graph);
  EXPECT_TRUE(stream_index_to_stream.count(side_stream.index) == 1);
  EXPECT_TRUE(stream_index_to_stream.size() == 2);
}

// --- edge / boundary ---

void should_capture_a_shared_input_once() {
  const array shared_input = ones({4});
  const array doubled = add(shared_input, shared_input);
  const CapturedGraph captured_graph = capture_graph({doubled, exp(shared_input)});
  int shared_input_count = 0;
  for (const auto& graph_node : captured_graph.nodes) {
    shared_input_count += graph_node.array_id == shared_input.id();
  }
  EXPECT_TRUE(shared_input_count == 1);
}

void should_capture_sibling_outputs_as_one_primitive() {
  const auto quotient_and_remainder = divmod(full({4}, 7.0f), full({4}, 3.0f));
  const array joined = add(quotient_and_remainder[0], quotient_and_remainder[1]);
  const CapturedGraph captured_graph = capture_graph({joined});
  int divmod_count = 0;
  for (const std::size_t node_index : captured_graph.topological) {
    divmod_count += captured_graph.nodes[node_index].op_name == "DivMod";
  }
  EXPECT_TRUE(divmod_count == 1);
}

// --- negative ---

void should_return_an_empty_graph_for_no_outputs() {
  const CapturedGraph captured_graph = capture_graph({});
  EXPECT_TRUE(captured_graph.nodes.empty());
  EXPECT_TRUE(captured_graph.topological.empty());
  EXPECT_TRUE(mlx_perf::collect_streams(captured_graph).empty());
}

void should_treat_an_evaluated_output_as_a_leaf_without_primitives() {
  array evaluated_output = exp(ones({4}));
  evaluated_output.eval();
  const CapturedGraph captured_graph = capture_graph({evaluated_output});
  EXPECT_TRUE(captured_graph.topological.empty());
  EXPECT_TRUE(captured_graph.nodes.size() == 1);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_order_every_primitive_after_its_primitive_inputs),
      TEST_CASE(should_label_edges_with_bytes_read_and_collect_streams),
      TEST_CASE(should_capture_a_shared_input_once),
      TEST_CASE(should_capture_sibling_outputs_as_one_primitive),
      TEST_CASE(should_return_an_empty_graph_for_no_outputs),
      TEST_CASE(should_treat_an_evaluated_output_as_a_leaf_without_primitives),
  });
}
