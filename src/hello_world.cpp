// The demo "job": a small two-branch computation built on two streams, so the
// report demonstrates Spark-style stage coloring (blue = stream 0, green =
// stream 1) and the cross-stream join edge. Orchestration only — every concern
// lives in its own module.

#include "graph_model.h"
#include "report_writer.h"
#include "telemetry.h"

#include "mlx/mlx.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using namespace mlx::core;
using Clock = std::chrono::steady_clock;

// Builds the same graph fresh so the uninstrumented pipeline pass measures a
// graph identical to the attributed pass, just without per-node syncs.
std::vector<array> build_model_arrays(
    const Stream& first_branch_stream,
    const Stream& second_branch_stream,
    std::vector<Stream>& captured_streams) {
  const Stream main_stream = default_stream(default_device());
  captured_streams = {main_stream, first_branch_stream, second_branch_stream};

  set_default_stream(main_stream);
  const array shared_input = random::normal({32, 64}, float32);

  // Branch one: an addmm whose matmul runs on stream 1.
  set_default_stream(first_branch_stream);
  const array first_branch_input = random::normal({32, 64}, float32);
  const array first_branch_weights = random::normal({64, 64}, float32);
  const array first_branch = addmm(
      shared_input, first_branch_input, first_branch_weights);

  // Branch two: a reduce+broadcast chain on stream 2, independent of branch
  // one so the scheduler may overlap the two streams.
  set_default_stream(second_branch_stream);
  const array second_branch_input = random::normal({32, 64}, float32);
  const array second_branch =
      arange(64, float32) * sum(second_branch_input, 0);

  // The join crosses streams back onto the main stream — the visible
  // cross-stage edge in the DAG.
  set_default_stream(main_stream);
  const array joined = add(first_branch, second_branch);
  const array activated = tanh(joined);
  const array normalized = softmax(activated, std::vector<int>{-1});
  return {normalized};
}

std::unordered_map<int, Stream> build_stream_index_to_stream(
    const std::vector<Stream>& captured_streams) {
  std::unordered_map<int, Stream> stream_index_to_stream;
  // emplace, not operator[]: Stream has no default constructor, so insertion
  // must construct it from the captured handle directly.
  for (const Stream& captured_stream : captured_streams) {
    stream_index_to_stream.emplace(captured_stream.index, captured_stream);
  }
  return stream_index_to_stream;
}

double elapsed_microseconds(Clock::time_point started_at) {
  return std::chrono::duration<double, std::micro>(Clock::now() - started_at)
      .count();
}

} // namespace

int main() {
  using namespace mlx_plan;
  const std::string output_directory = "build";
  const std::string report_title = "Hello World — two-stream toy model";

  std::printf("building the demo graph...\n");
  const Stream first_branch_stream = new_stream(default_device());
  const Stream second_branch_stream = new_stream(default_device());

  std::vector<Stream> pipeline_streams;
  std::vector<array> pipeline_output_nodes = build_model_arrays(
      first_branch_stream, second_branch_stream, pipeline_streams);
  const auto stream_index_to_stream =
      build_stream_index_to_stream(pipeline_streams);

  const CapturedGraph pipeline_graph = capture_graph(pipeline_output_nodes);
  std::printf(
      "captured %zu nodes (%zu primitives), %zu edges, %d streams\n",
      pipeline_graph.nodes.size(),
      pipeline_graph.topological.size(),
      pipeline_graph.edges.size(),
      pipeline_graph.stream_count);

  // Pass 1: the honest wall-clock. Evaluated once, synchronized once.
  std::printf("pass 1/2: pipeline pass (uninstrumented)...\n");
  const Clock::time_point pipeline_started_at = Clock::now();
  for (auto& output_node : pipeline_output_nodes) {
    output_node.eval();
  }
  synchronize(default_stream(default_device()));
  const double pipeline_microseconds = elapsed_microseconds(pipeline_started_at);

  // Pass 2: per-node attribution on a fresh, identical graph.
  std::printf("pass 2/2: attributed pass (node-by-node)...\n");
  std::vector<Stream> attributed_streams;
  std::vector<array> attributed_output_nodes = build_model_arrays(
      first_branch_stream, second_branch_stream, attributed_streams);
  const CapturedGraph attributed_graph = capture_graph(attributed_output_nodes);

  RunTelemetry run_telemetry =
      run_attributed_pass(attributed_graph, stream_index_to_stream);
  run_telemetry.pipeline_total_microseconds = pipeline_microseconds;

  const ReportPaths report_paths = write_report(
      attributed_graph,
      run_telemetry,
      output_directory,
      "hello_world",
      report_title);
  std::printf(
      "report: %s%s\n",
      report_paths.html_path.c_str(),
      report_paths.graphviz_available ? " (with SVG)" : " (no Graphviz found)");
  std::printf("done.\n");
  return 0;
}
