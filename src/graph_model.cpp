#include "graph_model.h"

#include "node_work_estimate.h"

// fast_primitives.h declares CustomKernel, whose state names a custom kernel;
// graph_utils.h supplies NodeNamer for the plan's variable names;
// primitives.h completes the Primitive type for op/stream access; utils.h
// declares the operator<< for Dtype/Shape that the label strings use.
#include "mlx/fast_primitives.h"
#include "mlx/graph_utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

#include <sstream>
#include <typeinfo>
#include <string_view>
#include <unordered_set>

namespace mlx_perf {

namespace {

using mlx::core::array;
using mlx::core::NodeNamer;

constexpr std::string_view CUSTOM_KERNEL_PRIMITIVE_NAME = "CustomKernel";

// Shape (as text and as integers), dtype and size: what a node's output is,
// whether it is an operation or data.
void describe_output(const array& node, GraphNode& captured_node) {
  captured_node.output_bytes = node.nbytes();
  std::ostringstream shape_text;
  shape_text << node.shape();
  captured_node.shape_string = shape_text.str();
  captured_node.shape_dimensions.assign(node.shape().begin(), node.shape().end());
  std::ostringstream dtype_text;
  dtype_text << node.dtype();
  captured_node.dtype_string = dtype_text.str();
}

// The author's name for a custom kernel (Metal or CUDA) lives in the
// primitive's state, not in name(). The class is checked first: a caller's
// own primitive may also call itself "CustomKernel" without being MLX's.
//
// Why typeid's name and not dynamic_cast: dynamic_cast needs MLX to export
// CustomKernel's typeinfo, which a shared MLX built with hidden visibility
// (its default, and the CUDA prefix's) does not. The dynamic type's name is
// read through the object's own vtable and needs no exported symbol, and the
// Itanium C++ ABI both macOS and Linux use mangles the class the same way.
constexpr const char* CUSTOM_KERNEL_MANGLED_CLASS_NAME = "N3mlx4core4fast12CustomKernelE";

std::string read_kernel_name(const mlx::core::Primitive& node_primitive) {
  if (CUSTOM_KERNEL_PRIMITIVE_NAME != node_primitive.name() ||
      std::string_view(typeid(node_primitive).name()) != CUSTOM_KERNEL_MANGLED_CLASS_NAME) {
    return "";
  }
  const auto& custom_kernel = static_cast<const mlx::core::fast::CustomKernel&>(node_primitive);
  return std::get<0>(custom_kernel.state());
}

GraphNode build_node(const array& node, NodeNamer& plan_namer) {
  GraphNode captured_node{.handle = node};
  captured_node.array_id = node.id();
  captured_node.label = plan_namer.get_name(node);
  captured_node.has_primitive = node.has_primitive();
  describe_output(node, captured_node);
  if (node.has_primitive()) {
    captured_node.op_name = node.primitive().name();
    captured_node.stream_index = node.primitive().stream().index;
    captured_node.kernel_name = read_kernel_name(node.primitive());
    captured_node.work = estimate_node_work(node);
  }
  return captured_node;
}

// Recursion carries the visit set explicitly so sibling outputs of one
// primitive (which share ids through the graph, not this struct) are each
// captured exactly once, matching MLX's own traversal semantics.
void visit_inputs(
    const array& node,
    NodeNamer& plan_namer,
    std::unordered_set<std::uintptr_t>& visited_ids,
    std::vector<std::size_t>& topological,
    std::vector<GraphNode>& captured_nodes) {
  if (visited_ids.find(node.id()) != visited_ids.end()) {
    return;
  }
  visited_ids.insert(node.id());
  for (const auto& sibling : node.siblings()) {
    visited_ids.insert(sibling.id());
  }
  for (const auto& input : node.inputs()) {
    visit_inputs(input, plan_namer, visited_ids, topological, captured_nodes);
  }
  captured_nodes.push_back(build_node(node, plan_namer));
  if (node.has_primitive()) {
    topological.push_back(captured_nodes.size() - 1);
  }
}

// Builds a leaf data node for an array that is a sibling output of some
// captured primitive: it carries data (shape/dtype/bytes) but shares the
// primitive with its siblings, so it renders as data, not as an op box.
GraphNode build_sibling_leaf_node(
    const array& sibling,
    const std::string& sibling_label) {
  GraphNode sibling_node{.handle = sibling};
  sibling_node.array_id = sibling.id();
  sibling_node.label = sibling_label;
  sibling_node.op_name = "(shared primitive output)";
  sibling_node.has_primitive = false;
  describe_output(sibling, sibling_node);
  return sibling_node;
}

std::vector<GraphEdge> capture_edges(CapturedGraph& captured_graph) {
  std::unordered_map<std::uintptr_t, std::size_t> node_id_to_index;
  for (std::size_t node_index = 0; node_index < captured_graph.nodes.size();
       ++node_index) {
    node_id_to_index.emplace(
        captured_graph.nodes[node_index].array_id, node_index);
  }

  std::vector<GraphEdge> captured_edges;
  for (std::size_t node_index = 0; node_index < captured_graph.nodes.size();
       ++node_index) {
    const GraphNode& captured_node = captured_graph.nodes[node_index];
    if (!captured_node.has_primitive) {
      continue;
    }
    for (const auto& input : captured_node.handle.inputs()) {
      const auto known_input = node_id_to_index.find(input.id());
      if (known_input == node_id_to_index.end()) {
        // Sibling output of another primitive: capture it as a leaf data
        // node so the edge resolves to a declared node.
        std::ostringstream sibling_label;
        sibling_label << "out" << captured_graph.nodes.size();
        captured_graph.nodes.push_back(
            build_sibling_leaf_node(input, sibling_label.str()));
        node_id_to_index.emplace(input.id(), captured_graph.nodes.size() - 1);
        captured_edges.push_back(
            {sibling_label.str(), captured_node.label, input.nbytes()});
        continue;
      }
      captured_edges.push_back(
          {captured_graph.nodes[known_input->second].label,
           captured_node.label,
           input.nbytes()});
    }
  }
  return captured_edges;
}

} // namespace

CapturedGraph capture_graph(const std::vector<array>& output_nodes) {
  CapturedGraph captured_graph;
  NodeNamer plan_namer;
  std::unordered_set<std::uintptr_t> visited_ids;
  for (const auto& output_node : output_nodes) {
    visit_inputs(
        output_node,
        plan_namer,
        visited_ids,
        captured_graph.topological,
        captured_graph.nodes);
  }
  captured_graph.edges = capture_edges(captured_graph);
  for (const GraphNode& captured_node : captured_graph.nodes) {
    captured_graph.stream_count = std::max(
        captured_graph.stream_count, captured_node.stream_index + 1);
  }
  return captured_graph;
}

std::unordered_map<int, mlx::core::Stream> collect_streams(
    const CapturedGraph& captured_graph) {
  std::unordered_map<int, mlx::core::Stream> stream_index_to_stream;
  for (const std::size_t node_index : captured_graph.topological) {
    const GraphNode& graph_node = captured_graph.nodes[node_index];
    if (graph_node.handle.has_primitive()) {
      // emplace: Stream has no default constructor.
      stream_index_to_stream.emplace(graph_node.stream_index,
                                     graph_node.handle.primitive().stream());
    }
  }
  return stream_index_to_stream;
}

} // namespace mlx_perf
