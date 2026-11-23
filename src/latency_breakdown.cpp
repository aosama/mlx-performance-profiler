#include "latency_breakdown.h"

#include <algorithm>

namespace mlx_plan {
namespace {

// Host-observed completion can only trail the true completion, never lead
// it, so a "ready" instant later than completion is observation jitter.
// Clamping every instant into the window keeps the slices non-negative and
// their sum exactly equal to the wall instead of inventing negative compute.
double clamp_into_window(
    double instant_microseconds,
    double window_start_microseconds,
    double window_end_microseconds) {
  return std::clamp(
      instant_microseconds, window_start_microseconds, window_end_microseconds);
}

// Advances the causal cursor to `blocked_until` (never backwards) and
// returns how long the node was blocked at this step.
double advance_cursor(double& cursor_microseconds, double blocked_until) {
  const double advanced_to = std::max(cursor_microseconds, blocked_until);
  const double blocked_microseconds = advanced_to - cursor_microseconds;
  cursor_microseconds = advanced_to;
  return blocked_microseconds;
}

} // namespace

std::string_view node_class_name(NodeClass node_class) {
  switch (node_class) {
    case NodeClass::compute:
      return "compute";
    case NodeClass::wait_on_cpu:
      return "wait_on_cpu";
    case NodeClass::wait_on_gpu:
      return "wait_on_gpu";
    case NodeClass::wait_on_dependency:
      return "wait_on_dependency";
  }
  return "unknown";
}

LatencyBreakdown decompose_node_latency(const NodeTimestamps& timestamps) {
  const double window_start = timestamps.issue_started_microseconds;
  const double window_end =
      std::max(window_start, timestamps.completed_microseconds);

  LatencyBreakdown breakdown;
  double cursor = window_start;
  breakdown.wait_on_cpu_microseconds = advance_cursor(
      cursor,
      clamp_into_window(
          timestamps.issue_finished_microseconds, window_start, window_end));
  breakdown.wait_on_gpu_microseconds = advance_cursor(
      cursor,
      clamp_into_window(
          timestamps.stream_ready_microseconds, window_start, window_end));
  breakdown.wait_on_dependency_microseconds = advance_cursor(
      cursor,
      clamp_into_window(
          timestamps.cross_stream_inputs_ready_microseconds,
          window_start,
          window_end));
  // The residual, exactly as StagePage derives Executor Computing Time.
  breakdown.compute_microseconds = window_end - cursor;
  breakdown.wall_microseconds = window_end - window_start;
  breakdown.classification = classify_breakdown(breakdown);
  return breakdown;
}

NodeClass classify_breakdown(const LatencyBreakdown& breakdown) {
  // Strict `>` against compute as the starting champion implements the
  // documented tie rule: equal waiting and working classifies as compute.
  NodeClass dominant_class = NodeClass::compute;
  double dominant_microseconds = breakdown.compute_microseconds;
  const NodeClass waiting_classes[] = {
      NodeClass::wait_on_cpu,
      NodeClass::wait_on_gpu,
      NodeClass::wait_on_dependency};
  const double waiting_microseconds[] = {
      breakdown.wait_on_cpu_microseconds,
      breakdown.wait_on_gpu_microseconds,
      breakdown.wait_on_dependency_microseconds};
  for (int slice_position = 0; slice_position < 3; ++slice_position) {
    if (waiting_microseconds[slice_position] > dominant_microseconds) {
      dominant_class = waiting_classes[slice_position];
      dominant_microseconds = waiting_microseconds[slice_position];
    }
  }
  return dominant_class;
}

} // namespace mlx_plan
