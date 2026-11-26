#pragma once

// A synthetic, GPU-free measurement for report tests: two streams, a
// producer on stream 1 feeding a join on stream 0 that stalled. Numbers are
// hand-picked so every slice is non-zero somewhere and slices sum to walls.

#include "gpu_timeline.h"
#include "graph_model.h"
#include "job_totals.h"
#include "latency_breakdown.h"
#include "paired_measurement.h"
#include "telemetry.h"

#include "mlx/array.h"

#include <string>

namespace report_fixture {

struct SyntheticRun {
  mlx_perf::CapturedGraph timeline_graph;
  mlx_perf::GpuTimeline gpu_timeline;
  mlx_perf::RunTelemetry serial_telemetry;
  mlx_perf::JobTotals job_totals;
};

inline mlx_perf::GraphNode make_graph_node(
    const std::string& label, const std::string& op_name, int stream_index) {
  // A scalar array is never evaluated here; it only gives the node a handle.
  mlx_perf::GraphNode graph_node{.handle = mlx::core::array(0.0f)};
  graph_node.array_id = graph_node.handle.id();
  graph_node.label = label;
  graph_node.op_name = op_name;
  graph_node.stream_index = stream_index;
  graph_node.shape_string = "(4,4)";
  graph_node.shape_dimensions = {4, 4};
  graph_node.dtype_string = "float32";
  graph_node.output_bytes = 64;
  graph_node.has_primitive = true;
  return graph_node;
}

inline mlx_perf::NodeTimestamps make_gpu_timestamps(
    double issue_started, double issue_finished, double stream_ready,
    double remote_ready, double gpu_started, double gpu_finished,
    double completed) {
  mlx_perf::NodeTimestamps timestamps;
  timestamps.issue_started_microseconds = issue_started;
  timestamps.issue_finished_microseconds = issue_finished;
  timestamps.stream_ready_microseconds = stream_ready;
  timestamps.cross_stream_inputs_ready_microseconds = remote_ready;
  timestamps.has_gpu_window = true;
  timestamps.gpu_started_microseconds = gpu_started;
  timestamps.gpu_finished_microseconds = gpu_finished;
  timestamps.completed_microseconds = completed;
  return timestamps;
}

// Must be called on a run that will not be copied afterwards: timeline nodes
// point into timeline_graph.nodes. Works for any struct with the same field
// names (SyntheticRun, mlx_perf::VariantMeasurement).
template <typename MeasuredRun>
void populate_two_stream_run(MeasuredRun& synthetic_run) {
  synthetic_run.timeline_graph.nodes = {
      make_graph_node("A", "MatMul", 1), make_graph_node("B<x>", "Add", 0)};
  synthetic_run.timeline_graph.topological = {0, 1};
  synthetic_run.timeline_graph.stream_count = 2;
  const mlx_perf::NodeTimestamps producer_timestamps =
      make_gpu_timestamps(0, 10, 0, 0, 15, 115, 130);
  const mlx_perf::NodeTimestamps join_timestamps =
      make_gpu_timestamps(12, 20, 0, 115, 118, 125, 140);
  const mlx_perf::NodeTimestamps all_timestamps[] = {
      producer_timestamps, join_timestamps};
  for (int position = 0; position < 2; ++position) {
    mlx_perf::NodeTimeline node_timeline;
    node_timeline.node = &synthetic_run.timeline_graph.nodes[position];
    node_timeline.timestamps = all_timestamps[position];
    node_timeline.breakdown =
        mlx_perf::decompose_node_latency(node_timeline.timestamps);
    synthetic_run.gpu_timeline.nodes.push_back(node_timeline);
  }
  synthetic_run.gpu_timeline.nodes[1].has_cross_stream_input = true;
  synthetic_run.gpu_timeline.nodes[1].cross_stream_producer_positions = {0};
  synthetic_run.gpu_timeline.streams = {
      {0, true, 1, 12, 140, 128, 7, 95}, {1, true, 1, 0, 130, 130, 100, 0}};
  synthetic_run.gpu_timeline.timeline_wall_microseconds = 140;
  synthetic_run.gpu_timeline.cross_stream_stall_microseconds = 95;
  synthetic_run.gpu_timeline.measured_gpu_compute_microseconds = 107;
  synthetic_run.gpu_timeline.gpu_measured_node_count = 2;
  synthetic_run.job_totals = {90, 80, 120, 7, 300, {}};
  synthetic_run.serial_telemetry = {};
}

// The same two nodes timed PER STAGE: "layer 0" (A, stream 1) feeds
// "layer 1" (B, stream 0) across streams, so the second task is a stalled
// join. Per-node timestamps stay zero, as a real per-stage pass leaves them.
template <typename MeasuredRun>
void populate_two_stage_run(MeasuredRun& synthetic_run) {
  populate_two_stream_run(synthetic_run);
  mlx_perf::GpuTimeline& gpu_timeline = synthetic_run.gpu_timeline;
  gpu_timeline.granularity = mlx_perf::TimelineGranularity::per_stage;
  const char* stage_names[] = {"layer 0", "layer 1"};
  for (int position = 0; position < 2; ++position) {
    mlx_perf::StageTimeline stage_task;
    stage_task.stage_index = position;
    stage_task.stage_name = stage_names[position];
    stage_task.stream_index = gpu_timeline.nodes[position].node->stream_index;
    stage_task.node_positions = {position};
    stage_task.timestamps = gpu_timeline.nodes[position].timestamps;
    stage_task.breakdown = gpu_timeline.nodes[position].breakdown;
    gpu_timeline.stages.push_back(stage_task);
    gpu_timeline.nodes[position].timestamps = {};
    gpu_timeline.nodes[position].breakdown = {};
    gpu_timeline.nodes[position].stage_position = position;
  }
  gpu_timeline.stages[1].has_cross_stream_input = true;
  gpu_timeline.stages[1].cross_stream_producer_stage_positions = {0};
}

// A paired measurement of two copies of the synthetic run, 12 rounds per
// metric: the candidate wins every GPU-span and host-submission round (a
// clear result) while its job-wall round differences go both ways, 6 of 12
// (not a result). Must not be copied after.
inline void populate_pair(mlx_perf::PairedMeasurement& paired_measurement) {
  populate_two_stream_run(paired_measurement.baseline);
  populate_two_stream_run(paired_measurement.candidate);
  paired_measurement.baseline.variant_name = "stock";
  paired_measurement.candidate.variant_name = "custom";
  paired_measurement.baseline_rounds = {
      {100, 104, 102, 101, 103, 99, 105, 100, 102, 101, 104, 103},
      {9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9},
      {50, 52, 51, 50, 52, 51, 50, 52, 51, 50, 52, 51}};
  paired_measurement.candidate_rounds = {
      {105, 94, 107, 98, 107, 93, 107, 99, 105, 99, 105, 99},
      {7, 7, 8, 7, 7, 8, 7, 7, 8, 7, 7, 8},
      {40, 42, 41, 40, 42, 41, 40, 42, 41, 40, 42, 41}};
  paired_measurement.output_agreement = {1, 0.25};
}

// The page without its <script> elements. Pages embed the graph view's
// libraries, whose code legitimately contains words such as "Infinity";
// checks for leaked "nan" or "inf" values belong to the rendered text.
inline std::string remove_script_elements(const std::string& document) {
  std::string visible_page;
  std::size_t copy_from = 0;
  for (std::size_t script_at = document.find("<script", copy_from); script_at != std::string::npos;
       script_at = document.find("<script", copy_from)) {
    visible_page += document.substr(copy_from, script_at - copy_from);
    const std::size_t script_end = document.find("</script>", script_at);
    copy_from = script_end == std::string::npos ? document.size() : script_end + 9;
  }
  return visible_page + document.substr(copy_from);
}

} // namespace report_fixture
