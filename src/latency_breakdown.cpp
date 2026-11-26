#include "latency_breakdown.h"

#include <algorithm>

namespace mlx_perf {
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
    case NodeClass::scheduler_delay:
      return "scheduler_delay";
  }
  return "unknown";
}

namespace {

// Host-only model (CPU streams, or no probe): compute is the residual.
LatencyBreakdown decompose_from_host_window(const NodeTimestamps& timestamps) {
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
  return breakdown;
}

// GPU-window model. Waits are bounded by the GPU window: stream and remote
// readiness that fall INSIDE [gpu_start, gpu_end] are an event wait encoded
// at the head of the buffer, so they are charged as waiting, not compute.
LatencyBreakdown decompose_with_gpu_window(const NodeTimestamps& timestamps) {
  const double window_start = timestamps.issue_started_microseconds;
  const double window_end =
      std::max(window_start, timestamps.completed_microseconds);
  const double gpu_started = clamp_into_window(
      timestamps.gpu_started_microseconds, window_start, window_end);
  const double gpu_finished = clamp_into_window(
      timestamps.gpu_finished_microseconds, gpu_started, window_end);

  LatencyBreakdown breakdown;
  breakdown.is_compute_measured = true;
  double cursor = window_start;
  // The buffer commits at the end of issue, so the GPU may start a hair
  // before the host stamps issue_finished; that overlap belongs to compute.
  breakdown.wait_on_cpu_microseconds = advance_cursor(
      cursor, std::min(timestamps.issue_finished_microseconds, gpu_started));
  breakdown.wait_on_gpu_microseconds = advance_cursor(
      cursor, std::min(timestamps.stream_ready_microseconds, gpu_finished));
  breakdown.wait_on_dependency_microseconds = advance_cursor(
      cursor,
      std::min(timestamps.cross_stream_inputs_ready_microseconds, gpu_finished));
  // Committed and unblocked, but the GPU had not begun the buffer yet:
  // driver/GPU submission, StagePage's Scheduler Delay.
  breakdown.scheduler_delay_microseconds = advance_cursor(cursor, gpu_started);
  breakdown.compute_microseconds = advance_cursor(cursor, gpu_finished);
  breakdown.completion_latency_microseconds = window_end - cursor;
  breakdown.wall_microseconds = window_end - window_start;
  return breakdown;
}

} // namespace

LatencyBreakdown decompose_node_latency(const NodeTimestamps& timestamps) {
  LatencyBreakdown breakdown = timestamps.has_gpu_window
      ? decompose_with_gpu_window(timestamps)
      : decompose_from_host_window(timestamps);
  breakdown.classification = classify_breakdown(breakdown);
  return breakdown;
}

NodeClass classify_breakdown(const LatencyBreakdown& breakdown) {
  // Strict `>` against compute as the starting champion implements the
  // documented tie rule: equal waiting and working classifies as compute.
  NodeClass dominant_class = NodeClass::compute;
  double dominant_microseconds = breakdown.compute_microseconds;
  constexpr int WAITING_CLASS_COUNT = 4;
  const NodeClass waiting_classes[WAITING_CLASS_COUNT] = {
      NodeClass::wait_on_cpu,
      NodeClass::wait_on_gpu,
      NodeClass::wait_on_dependency,
      NodeClass::scheduler_delay};
  const double waiting_microseconds[WAITING_CLASS_COUNT] = {
      breakdown.wait_on_cpu_microseconds,
      breakdown.wait_on_gpu_microseconds,
      breakdown.wait_on_dependency_microseconds,
      breakdown.scheduler_delay_microseconds};
  for (int slice_position = 0; slice_position < WAITING_CLASS_COUNT;
       ++slice_position) {
    if (waiting_microseconds[slice_position] > dominant_microseconds) {
      dominant_class = waiting_classes[slice_position];
      dominant_microseconds = waiting_microseconds[slice_position];
    }
  }
  return dominant_class;
}

} // namespace mlx_perf
