#include "node_work_estimate.h"

// fast_primitives.h declares ScaledDotProductAttention, whose state says
// whether attention is causal; ops.h supplies the flatten and astype that
// read a gathered multiply's indices.
#include "mlx/fast_primitives.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

#include <set>
#include <string_view>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>

namespace mlx_perf {
namespace {

using mlx::core::array;

// The matrix-multiply family: every one puts the shared (inner) dimension
// last on its first input, so its arithmetic is 2 × output elements × that
// dimension. QuantizedMatmul and GatherQMM take the full, unpacked width
// there; the packed weights are a later input.
const std::set<std::string_view> MATRIX_MULTIPLY_PRIMITIVE_NAMES = {
    "Matmul", "AddMM", "QuantizedMatmul", "GatherQMM", "GatherMM"};

// Broadcasts are views: the consuming kernel reads the source, so the
// source's size is what crosses memory, not the broadcast's logical size.
const std::set<std::string_view> BROADCAST_PRIMITIVE_NAMES = {"Broadcast", "BroadcastAxes"};

constexpr std::string_view ATTENTION_PRIMITIVE_NAME = "ScaledDotProductAttention";
// Checked by the mangled class name rather than dynamic_cast for the reason
// graph_model gives for CustomKernel: a shared MLX need not export typeinfo.
constexpr std::string_view ATTENTION_MANGLED_CLASS_NAME =
    "N3mlx4core4fast25ScaledDotProductAttentionE";

std::string_view primitive_name_of(const array& operation) {
  return operation.primitive().name();
}

std::uint64_t stored_bytes_of(const array& input) {
  if (input.has_primitive() && !input.inputs().empty() &&
      BROADCAST_PRIMITIVE_NAMES.contains(primitive_name_of(input))) {
    return stored_bytes_of(input.inputs()[0]);
  }
  return input.nbytes();
}

// An input listed twice (x · x) is read once.
std::uint64_t count_distinct_input_bytes(const array& operation) {
  std::unordered_set<std::uintptr_t> counted_input_ids;
  std::uint64_t input_bytes = 0;
  for (const array& input : operation.inputs()) {
    if (counted_input_ids.insert(input.id()).second) {
      input_bytes += stored_bytes_of(input);
    }
  }
  return input_bytes;
}

std::uint64_t count_output_bytes(const array& operation) {
  std::uint64_t output_bytes = operation.nbytes();
  for (const array& sibling : operation.siblings()) {
    output_bytes += sibling.nbytes();
  }
  return output_bytes;
}

double count_matrix_multiply_operations(const array& operation) {
  const array& left_operand = operation.inputs().front();
  if (left_operand.ndim() == 0) {
    return 0.0;
  }
  return 2.0 * static_cast<double>(operation.size()) *
      static_cast<double>(left_operand.shape(-1));
}

// Query-key pairs a causal mask keeps: query i (of L, at the end of S keys)
// sees S − L + i + 1 keys.
double count_attended_pairs(double query_length, double key_length, bool is_causal) {
  if (!is_causal || key_length < query_length) {
    return query_length * key_length;
  }
  return query_length * key_length - query_length * (query_length - 1.0) / 2.0;
}

// Scores (q · k) and the weighted sum of values, two operations per
// multiply-add each; the softmax between them is small beside the two.
bool count_attention_operations(const array& operation, NodeWork& node_work) {
  const mlx::core::Primitive& attention_primitive = operation.primitive();
  if (std::string_view(typeid(attention_primitive).name()) != ATTENTION_MANGLED_CLASS_NAME ||
      operation.inputs().size() < 3 || operation.inputs()[0].ndim() != 4) {
    return false;
  }
  const auto& attention = static_cast<const mlx::core::fast::ScaledDotProductAttention&>(
      attention_primitive);
  const bool is_causal = std::get<2>(attention.state());
  const array& queries = operation.inputs()[0];
  const array& keys = operation.inputs()[1];
  const array& values = operation.inputs()[2];
  const double query_heads = static_cast<double>(queries.shape(0)) * queries.shape(1);
  const double attended_pairs = count_attended_pairs(queries.shape(2), keys.shape(2), is_causal);
  node_work.arithmetic_operations =
      2.0 * query_heads * attended_pairs * (queries.shape(3) + values.shape(3));
  return true;
}

// The weights sit between the activations (first) and the two index arrays
// (last two); the expert count is the weights' leading extent.
void describe_gathered_weights(const array& operation, NodeWork& node_work) {
  const auto& inputs = operation.inputs();
  const array& expert_weights = inputs[1];
  if (inputs.size() < 4 || expert_weights.ndim() < 3) {
    return;
  }
  node_work.is_gathered = true;
  node_work.gathered_experts_total = static_cast<int>(
      expert_weights.size() / (static_cast<std::size_t>(expert_weights.shape(-1)) *
                               static_cast<std::size_t>(expert_weights.shape(-2))));
  for (std::size_t weight_index = 1; weight_index + 2 < inputs.size(); ++weight_index) {
    node_work.gathered_weight_bytes_all_experts += stored_bytes_of(inputs[weight_index]);
  }
  node_work.gathered_index_array_id = inputs.back().id();
}

int count_distinct_experts(const array& expert_indices) {
  array flat_indices = mlx::core::astype(mlx::core::flatten(expert_indices), mlx::core::uint32);
  flat_indices.eval();
  const std::uint32_t* index_values = flat_indices.data<std::uint32_t>();
  std::unordered_set<std::uint32_t> used_experts(index_values, index_values + flat_indices.size());
  return static_cast<int>(used_experts.size());
}

// Swaps the every-expert weight bytes for the used experts' share. Experts
// are equal in size, so the share is proportional.
void narrow_to_used_experts(NodeWork& node_work, int used_expert_count) {
  if (node_work.gathered_experts_total <= 0 || used_expert_count <= 0) {
    return;
  }
  const std::uint64_t used_weight_bytes = node_work.gathered_weight_bytes_all_experts *
      static_cast<std::uint64_t>(used_expert_count) /
      static_cast<std::uint64_t>(node_work.gathered_experts_total);
  node_work.bytes_read -= node_work.gathered_weight_bytes_all_experts - used_weight_bytes;
  node_work.gathered_experts_used = used_expert_count;
}

} // namespace

NodeWork estimate_node_work(const array& operation) {
  NodeWork node_work;
  if (!operation.has_primitive()) {
    return node_work;
  }
  node_work.bytes_read = count_distinct_input_bytes(operation);
  node_work.bytes_written = count_output_bytes(operation);
  const std::string_view primitive_name = primitive_name_of(operation);
  if (MATRIX_MULTIPLY_PRIMITIVE_NAMES.contains(primitive_name) && !operation.inputs().empty()) {
    node_work.is_arithmetic_counted = true;
    node_work.arithmetic_operations = count_matrix_multiply_operations(operation);
  } else if (primitive_name == ATTENTION_PRIMITIVE_NAME) {
    node_work.is_arithmetic_counted = count_attention_operations(operation, node_work);
  }
  if (primitive_name == "GatherQMM" || primitive_name == "GatherMM") {
    describe_gathered_weights(operation, node_work);
  }
  return node_work;
}

void resolve_gathered_experts(CapturedGraph& captured_graph) {
  std::unordered_map<std::uintptr_t, const GraphNode*> array_id_to_node;
  for (const GraphNode& graph_node : captured_graph.nodes) {
    array_id_to_node.emplace(graph_node.array_id, &graph_node);
  }
  for (GraphNode& graph_node : captured_graph.nodes) {
    NodeWork& node_work = graph_node.work;
    if (!node_work.is_gathered || node_work.gathered_experts_used > 0) {
      continue;
    }
    const auto index_node = array_id_to_node.find(node_work.gathered_index_array_id);
    if (index_node == array_id_to_node.end() || !index_node->second->handle.is_available()) {
      continue;
    }
    narrow_to_used_experts(node_work, count_distinct_experts(index_node->second->handle));
  }
}

} // namespace mlx_perf
