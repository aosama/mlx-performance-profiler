#include "timeline_console.h"

#include <array>
#include <cstdio>

namespace mlx_plan {
namespace {

constexpr double MICROSECONDS_PER_MILLISECOND = 1000.0;

double to_milliseconds(double microseconds) {
  return microseconds / MICROSECONDS_PER_MILLISECOND;
}

void print_class_counts(const GpuTimeline& gpu_timeline) {
  const std::array<NodeClass, 4> all_classes = {
      NodeClass::compute,
      NodeClass::wait_on_cpu,
      NodeClass::wait_on_gpu,
      NodeClass::wait_on_dependency};
  std::printf("  node classes            :");
  for (const NodeClass node_class : all_classes) {
    int class_count = 0;
    for (const NodeTimeline& node_timeline : gpu_timeline.nodes) {
      class_count += node_timeline.breakdown.classification == node_class;
    }
    std::printf(
        " %.*s %d ",
        static_cast<int>(node_class_name(node_class).size()),
        node_class_name(node_class).data(),
        class_count);
  }
  std::printf("\n");
}

} // namespace

void print_job_summary(
    const JobTotals& job_totals, const GpuTimeline& gpu_timeline) {
  const double instrumentation_overhead_microseconds =
      gpu_timeline.timeline_wall_microseconds - job_totals.job_wall_microseconds;
  std::printf("\n== job summary ==\n");
  std::printf(
      "  job wall                : %8.3f ms  single warm eval, streams overlap"
      " (the honest end-to-end number)\n",
      to_milliseconds(job_totals.job_wall_microseconds));
  std::printf(
      "  timeline wall           : %8.3f ms  per-node async_eval pass; %+.3f ms"
      " vs job wall is instrumentation (one command buffer per node)\n",
      to_milliseconds(gpu_timeline.timeline_wall_microseconds),
      to_milliseconds(instrumentation_overhead_microseconds));
  std::printf(
      "  sum-of-stages           : %8.3f ms  serial pass, each node synchronized"
      " alone (no overlap; Spark's summed stage durations)\n",
      to_milliseconds(job_totals.sum_of_stages_microseconds));
  std::printf(
      "  cross-stream stall      : %8.3f ms  summed wait_on_dependency, Spark"
      " fetchWaitTime analog (stream-time; can exceed wall)\n",
      to_milliseconds(gpu_timeline.cross_stream_stall_microseconds));
  print_class_counts(gpu_timeline);
}

void print_stream_table(const GpuTimeline& gpu_timeline) {
  std::printf("\n== streams (timeline pass) ==\n");
  std::printf(
      "  %-6s %-6s %5s %10s %10s %12s %10s\n",
      "stream", "device", "nodes", "wall ms", "busy ms", "dep-wait ms",
      "idle ms");
  for (const StreamTimeline& stream_timeline : gpu_timeline.streams) {
    // In an in-order stream compute residuals cannot overlap, so whatever
    // of the wall is not busy was spent waiting or with nothing issued.
    const double idle_microseconds =
        stream_timeline.wall_microseconds - stream_timeline.busy_microseconds;
    std::printf(
        "  %-6d %-6s %5d %10.3f %10.3f %12.3f %10.3f\n",
        stream_timeline.stream_index,
        stream_timeline.is_gpu ? "gpu" : "cpu",
        stream_timeline.node_count,
        to_milliseconds(stream_timeline.wall_microseconds),
        to_milliseconds(stream_timeline.busy_microseconds),
        to_milliseconds(stream_timeline.dependency_wait_microseconds),
        to_milliseconds(idle_microseconds));
  }
}

void print_node_table(const GpuTimeline& gpu_timeline) {
  std::printf("\n== nodes (timeline pass; µs, host-observed) ==\n");
  std::printf(
      "  %4s %-6s %-22s %3s %4s %9s %8s %8s %8s %8s %8s  %s\n",
      "#", "node", "op", "str", "join", "issued@", "cpu", "gpu-q", "dep",
      "compute", "wall", "class");
  int dispatch_position = 0;
  for (const NodeTimeline& node_timeline : gpu_timeline.nodes) {
    const LatencyBreakdown& breakdown = node_timeline.breakdown;
    const std::string_view class_name =
        node_class_name(breakdown.classification);
    std::printf(
        "  %4d %-6s %-22.22s %3d %4s %9.1f %8.1f %8.1f %8.1f %8.1f %8.1f  %.*s\n",
        ++dispatch_position,
        node_timeline.node->label.c_str(),
        node_timeline.node->op_name.c_str(),
        node_timeline.node->stream_index,
        node_timeline.has_cross_stream_input ? "yes" : "",
        node_timeline.timestamps.issue_started_microseconds,
        breakdown.wait_on_cpu_microseconds,
        breakdown.wait_on_gpu_microseconds,
        breakdown.wait_on_dependency_microseconds,
        breakdown.compute_microseconds,
        breakdown.wall_microseconds,
        static_cast<int>(class_name.size()),
        class_name.data());
  }
  std::printf(
      "  limits: completion is host-observed (trails kernel end by event"
      " signal + poll gap); compute is the window residual, so for small\n"
      "  kernels it is mostly the per-node command-buffer round trip, not"
      " kernel ns — kernel-accurate GPU time arrives with Metal capture"
      " (Phase 2).\n");
}

} // namespace mlx_plan
