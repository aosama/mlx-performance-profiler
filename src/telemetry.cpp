#include "telemetry.h"

#include "mlx/memory.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace mlx_plan {
namespace {

using mlx::core::Stream;
using Clock = std::chrono::steady_clock;

double elapsed_microseconds(Clock::time_point started_at) {
  return std::chrono::duration<double, std::micro>(
             Clock::now() - started_at)
      .count();
}

// Live progress with an ETA, per repo convention: any run of non-trivial
// length reports where it is, not just where it ended up.
void print_node_progress(
    int node_position,
    int node_count,
    const std::string& node_name,
    double elapsed_total_seconds,
    double last_duration_seconds) {
  const double nodes_per_second =
      last_duration_seconds > 0.0 ? 1.0 / last_duration_seconds : 0.0;
  const double remaining_seconds =
      nodes_per_second > 0.0
          ? (node_count - node_position) / nodes_per_second
          : 0.0;
  std::printf(
      "  node %d/%d (%s)  elapsed %.1fs  eta %.1fs\n",
      node_position,
      node_count,
      node_name.c_str(),
      elapsed_total_seconds,
      remaining_seconds);
}

} // namespace

// Runs one node: reset the memory ceiling, evaluate on the node's stream, then
// synchronize that same stream so wall-clock includes the GPU wait — the
// serialize-for-measurement tradeoff documented in PLAN.md.
NodeTelemetry measure_node(
    const GraphNode& captured_node,
    const Stream& execution_stream) {
  NodeTelemetry node_telemetry;
  node_telemetry.node = &captured_node;
  node_telemetry.output_bytes = captured_node.output_bytes;

  const size_t active_bytes_before = mlx::core::get_active_memory();
  mlx::core::reset_peak_memory();

  const Clock::time_point started_at = Clock::now();
  // An array copy shares the same underlying descriptor, so evaluating the
  // copy schedules this node's compute — array::eval() itself is non-const.
  mlx::core::array node_handle = captured_node.handle;
  node_handle.eval();
  mlx::core::synchronize(execution_stream);
  node_telemetry.duration_microseconds = elapsed_microseconds(started_at);

  node_telemetry.memory_added_bytes =
      mlx::core::get_active_memory() - active_bytes_before;
  return node_telemetry;
}

RunTelemetry run_attributed_pass(
    const CapturedGraph& captured_graph,
    const std::unordered_map<int, Stream>& stream_index_to_stream) {
  RunTelemetry run_telemetry;
  run_telemetry.node_count =
      static_cast<int>(captured_graph.topological.size());

  std::printf(
      "attributed pass: timing %d nodes individually\n", run_telemetry.node_count);
  const Clock::time_point pass_started_at = Clock::now();

  for (const std::size_t node_index : captured_graph.topological) {
    const GraphNode& captured_node = captured_graph.nodes[node_index];
    const Stream execution_stream =
        stream_index_to_stream.at(captured_node.stream_index);
    NodeTelemetry node_telemetry = measure_node(captured_node, execution_stream);
    run_telemetry.attributed_nodes.push_back(node_telemetry);
    run_telemetry.attributed_total_microseconds +=
        node_telemetry.duration_microseconds;
    run_telemetry.run_peak_memory_bytes = std::max(
        run_telemetry.run_peak_memory_bytes, mlx::core::get_peak_memory());

    print_node_progress(
        static_cast<int>(run_telemetry.attributed_nodes.size()),
        run_telemetry.node_count,
        captured_node.op_name.empty() ? captured_node.label
                                      : captured_node.op_name + " " +
                                            captured_node.label,
        elapsed_microseconds(pass_started_at) / 1e6,
        node_telemetry.duration_microseconds / 1e6);
  }
  return run_telemetry;
}

} // namespace mlx_plan
