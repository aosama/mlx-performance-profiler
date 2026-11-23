// The demo "job": a small two-branch computation on three streams — the main
// stream (index 0) feeds two side-stream branches (indices 1 and 2) that join
// back on the main stream — so the report shows Spark-style per-stream stage
// coloring and a real cross-stream join. Orchestration only — every concern
// lives in its own module.

#include "gpu_timeline.h"
#include "graph_model.h"
#include "report_writer.h"
#include "telemetry.h"
#include "timeline_console.h"

#include "mlx/mlx.h"

#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using namespace mlx::core;

// Builds the same graph fresh on every call, so the warm-up, pipeline,
// timeline and attributed passes each measure an identical graph.
std::vector<array> build_model_arrays(
    const Stream& first_branch_stream,
    const Stream& second_branch_stream) {
  const Stream main_stream = default_stream(default_device());

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

struct DemoStreams {
  Stream first_branch_stream;
  Stream second_branch_stream;
  std::unordered_map<int, Stream> stream_index_to_stream;
};

// MLX creates the default stream lazily and numbers streams in creation
// order (stream.cpp), so the main stream is touched FIRST to make it index 0
// — the index the report's stage coloring and this file's header assume.
DemoStreams make_demo_streams() {
  const Stream main_stream = default_stream(default_device());
  const Stream first_branch_stream = new_stream(default_device());
  const Stream second_branch_stream = new_stream(default_device());
  return {
      first_branch_stream,
      second_branch_stream,
      build_stream_index_to_stream(
          {main_stream, first_branch_stream, second_branch_stream})};
}

// Every pass gets its own freshly built graph: evaluation materializes and
// detaches nodes, so a graph can be measured exactly once.
std::vector<array> build_fresh_graph(const DemoStreams& demo_streams) {
  return build_model_arrays(
      demo_streams.first_branch_stream, demo_streams.second_branch_stream);
}

mlx_plan::CapturedGraph capture_fresh_graph(const DemoStreams& demo_streams) {
  mlx_plan::CapturedGraph fresh_graph =
      mlx_plan::capture_graph(build_fresh_graph(demo_streams));
  std::printf(
      "  captured %zu nodes (%zu primitives), %zu edges, %d streams\n",
      fresh_graph.nodes.size(),
      fresh_graph.topological.size(),
      fresh_graph.edges.size(),
      fresh_graph.stream_count);
  return fresh_graph;
}

void write_html_report(
    const mlx_plan::CapturedGraph& timeline_graph,
    const mlx_plan::GpuTimeline& gpu_timeline,
    const mlx_plan::RunTelemetry& serial_telemetry,
    const mlx_plan::JobTotals& job_totals) {
  const mlx_plan::ReportInputs report_inputs{
      "Hello World — two-stream toy model", timeline_graph, gpu_timeline,
      serial_telemetry, job_totals};
  const mlx_plan::ReportPaths report_paths =
      mlx_plan::write_report(report_inputs, "build", "hello_world");
  std::printf(
      "\nreport: %s%s\n",
      report_paths.html_path.c_str(),
      report_paths.graphviz_available ? " (with SVG)" : " (no Graphviz found)");
}

} // namespace

int main() {
  using namespace mlx_plan;
  const DemoStreams demo_streams = make_demo_streams();

  // Pass 0 is discarded: the first eval in a process pays Metal library
  // load and kernel compilation, which would otherwise inflate the job wall.
  std::printf("pass 0/3: warm-up (kernel compile; discarded)...\n");
  run_pipeline_pass(build_fresh_graph(demo_streams));

  std::printf("pass 1/3: pipeline pass (uninstrumented job wall)...\n");
  JobTotals job_totals;
  job_totals.job_wall_microseconds =
      run_pipeline_pass(build_fresh_graph(demo_streams));

  std::printf("pass 2/3: timeline pass (per-node async_eval, overlap kept)...\n");
  const CapturedGraph timeline_graph = capture_fresh_graph(demo_streams);
  const GpuTimeline gpu_timeline = capture_gpu_timeline(
      timeline_graph, demo_streams.stream_index_to_stream);

  std::printf("pass 3/3: attributed pass (serial, sum-of-stages)...\n");
  const CapturedGraph attributed_graph = capture_fresh_graph(demo_streams);
  const RunTelemetry run_telemetry = run_attributed_pass(
      attributed_graph, demo_streams.stream_index_to_stream);
  job_totals.sum_of_stages_microseconds =
      run_telemetry.attributed_total_microseconds;

  print_job_summary(job_totals, gpu_timeline);
  print_stream_table(gpu_timeline);
  print_node_table(gpu_timeline);
  write_html_report(timeline_graph, gpu_timeline, run_telemetry, job_totals);
  return 0;
}
