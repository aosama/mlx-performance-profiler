// Integration tests for node_work_estimate: bytes and arithmetic worked out
// from real MLX operations, and a gathered multiply narrowed to the experts
// its indices used (which evaluates on the GPU).

#include "graph_model.h"
#include "node_work_estimate.h"
#include "test_harness.h"

#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include "mlx/version.h"

#include <cstdio>
#include <string_view>
#include <vector>

namespace {

using namespace mlx_perf;
namespace mx = mlx::core;

// --- positive ---

void should_count_a_matrix_multiply_from_its_shapes() {
  const mx::array left_matrix = mx::ones({64, 128});
  const mx::array right_matrix = mx::ones({128, 32});
  const NodeWork node_work = estimate_node_work(mx::matmul(left_matrix, right_matrix));
  EXPECT_TRUE(node_work.bytes_read == (64 * 128 + 128 * 32) * 4);
  EXPECT_TRUE(node_work.bytes_written == 64 * 32 * 4);
  EXPECT_TRUE(node_work.is_arithmetic_counted);
  EXPECT_NEAR(node_work.arithmetic_operations, 2.0 * 64 * 32 * 128, 1e-6);
}

void should_count_quantized_weights_as_packed() {
  std::vector<mx::array> quantized = mx::quantize(mx::ones({256, 128}), 64, 4);
  mx::eval(quantized);
  const mx::array activations = mx::ones({8, 128});
  const NodeWork node_work = estimate_node_work(
      mx::quantized_matmul(activations, quantized[0], quantized[1], quantized[2], true, 64, 4));
  const std::uint64_t packed_weight_bytes =
      quantized[0].nbytes() + quantized[1].nbytes() + quantized[2].nbytes();
  EXPECT_TRUE(node_work.bytes_read == activations.nbytes() + packed_weight_bytes);
  EXPECT_TRUE(packed_weight_bytes < 256 * 128 * 4 / 4);
  EXPECT_NEAR(node_work.arithmetic_operations, 2.0 * 8 * 256 * 128, 1e-6);
}

void should_count_fused_attention_arithmetic() {
  const mx::array queries = mx::ones({1, 4, 32, 64}, mx::float16);
  const mx::array keys = mx::ones({1, 2, 32, 64}, mx::float16);
  const mx::array values = mx::ones({1, 2, 32, 64}, mx::float16);
  const mx::array attention_output = mx::fast::scaled_dot_product_attention(queries, keys, values, 0.125f);
  if (std::string_view(attention_output.primitive().name()) != "ScaledDotProductAttention") {
    std::printf("    note: MLX chose its unfused attention here; nothing fused to count\n");
    return;
  }
  const NodeWork node_work = estimate_node_work(attention_output);
  EXPECT_TRUE(node_work.is_arithmetic_counted);
  EXPECT_NEAR(node_work.arithmetic_operations, 2.0 * 4 * 32 * 32 * (64 + 64), 1e-6);
}

void should_narrow_a_gathered_multiply_to_the_experts_it_used() {
  std::vector<mx::array> quantized = mx::quantize(mx::ones({8, 64, 128}), 64, 4);
  mx::eval(quantized);
  const mx::array chosen_experts = mx::array({1u, 1u, 3u, 5u}, mx::uint32);
  const mx::array gathered_output = mx::gather_qmm(mx::ones({4, 1, 128}), quantized[0], quantized[1],
                                                   quantized[2], std::nullopt, chosen_experts, true, 64, 4);
  CapturedGraph captured_graph = capture_graph({gathered_output});
  const GraphNode& gather_node = captured_graph.nodes[captured_graph.topological.back()];
  const std::uint64_t every_expert_bytes_read = gather_node.work.bytes_read;
  EXPECT_TRUE(gather_node.work.is_gathered && gather_node.work.gathered_experts_total == 8);
  EXPECT_NEAR(gather_node.work.arithmetic_operations, 2.0 * 4 * 1 * 64 * 128, 1e-6);
  mx::eval({gathered_output});
  resolve_gathered_experts(captured_graph);
  const NodeWork& resolved_work = captured_graph.nodes[captured_graph.topological.back()].work;
  const std::uint64_t all_weight_bytes = resolved_work.gathered_weight_bytes_all_experts;
  EXPECT_TRUE(resolved_work.gathered_experts_used == 3);
  EXPECT_TRUE(every_expert_bytes_read - resolved_work.bytes_read == all_weight_bytes - all_weight_bytes * 3 / 8);
}

void should_narrow_an_unquantized_gathered_multiply_to_its_experts() {
  const mx::array expert_weights = mx::ones({4, 128, 32});
  const mx::array chosen_experts = mx::array({0u, 2u, 2u}, mx::uint32);
  const mx::array gathered_output =
      mx::gather_mm(mx::ones({3, 1, 128}), expert_weights, std::nullopt, chosen_experts);
  CapturedGraph captured_graph = capture_graph({gathered_output});
  EXPECT_NEAR(captured_graph.nodes[captured_graph.topological.back()].work.arithmetic_operations,
              2.0 * 3 * 1 * 32 * 128, 1e-6);
  mx::eval({gathered_output});
  resolve_gathered_experts(captured_graph);
  const NodeWork& resolved_work = captured_graph.nodes[captured_graph.topological.back()].work;
  EXPECT_TRUE(resolved_work.gathered_experts_total == 4 && resolved_work.gathered_experts_used == 2);
  EXPECT_TRUE(resolved_work.gathered_weight_bytes_all_experts == expert_weights.nbytes());
}

// --- edge / boundary ---

void should_read_an_input_listed_twice_once() {
  const mx::array squared_values = mx::ones({64, 128});
  const NodeWork node_work = estimate_node_work(mx::multiply(squared_values, squared_values));
  EXPECT_TRUE(node_work.bytes_read == squared_values.nbytes());
}

void should_count_a_broadcast_input_at_its_source_size() {
  const mx::array activations = mx::ones({64, 128});
  const mx::array bias = mx::ones({128});
  const NodeWork node_work = estimate_node_work(mx::add(activations, bias));
  EXPECT_TRUE(node_work.bytes_read == activations.nbytes() + bias.nbytes());
}

void should_count_causal_attention_on_the_kept_pairs_only() {
  const mx::array queries = mx::ones({1, 4, 32, 64}, mx::float16);
  const mx::array keys = mx::ones({1, 2, 32, 64}, mx::float16);
#if MLX_VERSION_MAJOR == 0 && MLX_VERSION_MINOR == 32 && MLX_VERSION_PATCH < 5
  const mx::array attention_output =
      mx::fast::scaled_dot_product_attention(queries, keys, keys, 0.125f, "causal");
#else
  const mx::array attention_output =
      mx::fast::scaled_dot_product_attention(queries, keys, keys, 0.125f, -1, "causal");
#endif
  if (std::string_view(attention_output.primitive().name()) != "ScaledDotProductAttention") {
    std::printf("    note: MLX chose its unfused attention here; nothing fused to count\n");
    return;
  }
  const double kept_pairs = 32.0 * 32.0 - 32.0 * 31.0 / 2.0;
  EXPECT_NEAR(estimate_node_work(attention_output).arithmetic_operations, 2.0 * 4 * kept_pairs * 128, 1e-6);
}

// --- negative ---

void should_leave_elementwise_arithmetic_uncounted() {
  const mx::array activations = mx::ones({64, 128});
  const NodeWork node_work = estimate_node_work(mx::exp(activations));
  EXPECT_TRUE(!node_work.is_arithmetic_counted);
  EXPECT_TRUE(node_work.bytes_read == activations.nbytes() && node_work.bytes_written == activations.nbytes());
}

void should_give_an_input_leaf_no_work() {
  const NodeWork node_work = estimate_node_work(mx::array({1.0f, 2.0f, 3.0f, 4.0f}));
  EXPECT_TRUE(node_work.bytes_read == 0 && node_work.bytes_written == 0);
}

void should_count_every_expert_when_the_indices_never_materialized() {
  std::vector<mx::array> quantized = mx::quantize(mx::ones({8, 64, 128}), 64, 4);
  mx::eval(quantized);
  const mx::array chosen_experts = mx::array({2u, 6u}, mx::uint32);
  CapturedGraph captured_graph = capture_graph({mx::gather_qmm(
      mx::ones({2, 1, 128}), quantized[0], quantized[1], quantized[2], std::nullopt,
      mx::add(chosen_experts, mx::array(0u, mx::uint32)), true, 64, 4)});
  const std::uint64_t every_expert_bytes_read =
      captured_graph.nodes[captured_graph.topological.back()].work.bytes_read;
  resolve_gathered_experts(captured_graph);  // nothing evaluated yet
  const NodeWork& unresolved_work = captured_graph.nodes[captured_graph.topological.back()].work;
  EXPECT_TRUE(unresolved_work.gathered_experts_used == 0);
  EXPECT_TRUE(unresolved_work.bytes_read == every_expert_bytes_read);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_count_a_matrix_multiply_from_its_shapes),
      TEST_CASE(should_count_quantized_weights_as_packed),
      TEST_CASE(should_count_fused_attention_arithmetic),
      TEST_CASE(should_narrow_a_gathered_multiply_to_the_experts_it_used),
      TEST_CASE(should_narrow_an_unquantized_gathered_multiply_to_its_experts),
      TEST_CASE(should_read_an_input_listed_twice_once),
      TEST_CASE(should_count_a_broadcast_input_at_its_source_size),
      TEST_CASE(should_count_causal_attention_on_the_kept_pairs_only),
      TEST_CASE(should_leave_elementwise_arithmetic_uncounted),
      TEST_CASE(should_give_an_input_leaf_no_work),
      TEST_CASE(should_count_every_expert_when_the_indices_never_materialized),
  });
}
