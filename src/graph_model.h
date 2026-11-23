#pragma once

// Graph capture: turns a set of MLX output arrays into the structured "plan"
// every other module consumes. One concern only: walking the graph.

#include "mlx/array.h"
#include "mlx/stream.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_plan {

// One captured node is one MLX array. Arrays with a primitive are operations
// (the Spark "task"/"stage" unit); arrays without one are user inputs that
// already hold data. Handles are kept by value so the graph stays materialized
// while telemetry evaluates it node by node.
struct GraphNode {
  mlx::core::array handle;
  std::uintptr_t array_id = 0;
  std::string label;           // plan variable name (A, B, C… from NodeNamer)
  std::string op_name;         // e.g. "MatMul"; empty for input leaves
  int stream_index = 0;        // the stream this primitive executes on
  std::string shape_string;    // "[512, 256]"
  std::string dtype_string;    // "float32"
  size_t output_bytes = 0;     // materialization size of this node's output
  bool has_primitive = false;  // false only for input leaves
};

// A data-dependency edge in the Spark "shuffle bytes" sense: the consumer
// reads `bytes` of input from the producer.
struct GraphEdge {
  std::string from_label;
  std::string to_label;
  size_t bytes = 0;
};

struct CapturedGraph {
  std::vector<GraphNode> nodes;
  std::vector<GraphEdge> edges;
  // Indices into nodes — primitives, inputs before consumers (MLX's own
  // post-order tape order, which is the order per-node telemetry executes
  // them in). Indices, not pointers: the nodes vector reallocates while the
  // capture fills it, so pointers taken during capture would dangle.
  std::vector<std::size_t> topological;
  int stream_count = 0;
  size_t total_input_bytes = 0;
};

// Walks the graph depth-first from `output_nodes`, mirroring MLX's own
// depth_first_traversal so the capture cannot drift from what MLX executes.
CapturedGraph capture_graph(const std::vector<mlx::core::array>& output_nodes);

// The Stream handles the graph's primitives run on, keyed by index — read
// from the primitives themselves, so callers that did not create the streams
// (e.g. through the C API) need not supply them. Call before evaluation:
// evaluated nodes drop their primitive.
std::unordered_map<int, mlx::core::Stream> collect_streams(
    const CapturedGraph& captured_graph);

} // namespace mlx_plan
