#include "timeline_console.h"

#include "gpu_backend_vocabulary.h"
#include "timed_units.h"

#include <array>
#include <cstdio>
#include <string>

namespace mlx_perf {
namespace {

constexpr double MICROSECONDS_PER_MILLISECOND = 1000.0;

double to_milliseconds(double microseconds) {
  return microseconds / MICROSECONDS_PER_MILLISECOND;
}

std::string format_gpu_start(const NodeTimestamps& timestamps) {
  if (!timestamps.has_gpu_window) {
    return "-";
  }
  char formatted_start[32];
  std::snprintf(
      formatted_start, sizeof(formatted_start), "%.1f",
      timestamps.gpu_started_microseconds);
  return formatted_start;
}

void print_class_counts(const GpuTimeline& gpu_timeline) {
  const std::array<NodeClass, 5> all_classes = {
      NodeClass::compute,
      NodeClass::wait_on_cpu,
      NodeClass::wait_on_gpu,
      NodeClass::wait_on_dependency,
      NodeClass::scheduler_delay};
  const std::vector<TimedUnit> timed_units = collect_timed_units(gpu_timeline);
  std::printf("  %-5s classes           :", timed_unit_noun(gpu_timeline));
  for (const NodeClass node_class : all_classes) {
    int class_count = 0;
    for (const TimedUnit& timed_unit : timed_units) {
      class_count += timed_unit.breakdown->classification == node_class;
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
      "  job wall                : %8.3f ms  median of %d warm evals (range"
      " %.3f–%.3f ms), streams overlap: the honest end-to-end number\n",
      to_milliseconds(job_totals.job_wall_microseconds),
      job_totals.job_wall_sample_count,
      to_milliseconds(job_totals.job_wall_min_microseconds),
      to_milliseconds(job_totals.job_wall_max_microseconds));
  const char* unit_noun = timed_unit_noun(gpu_timeline);
  if (is_within_job_wall_range(job_totals, gpu_timeline.timeline_wall_microseconds)) {
    std::printf(
        "  timeline wall           : %8.3f ms  per-%s async_eval pass; inside the"
        " job-wall range, so no measurable instrumentation cost\n",
        to_milliseconds(gpu_timeline.timeline_wall_microseconds), unit_noun);
  } else {
    std::printf(
        "  timeline wall           : %8.3f ms  per-%s async_eval pass; %+.3f ms"
        " vs job wall is instrumentation (at least one command buffer per %s)\n",
        to_milliseconds(gpu_timeline.timeline_wall_microseconds), unit_noun,
        to_milliseconds(instrumentation_overhead_microseconds), unit_noun);
  }
  std::printf(
      "  sum-of-stages           : %8.3f ms  serial pass, each node synchronized"
      " alone (no overlap; Spark's summed stage durations)\n",
      to_milliseconds(job_totals.sum_of_stages_microseconds));
  std::printf(
      "  cross-stream stall      : %8.3f ms  summed wait_on_dependency, Spark"
      " fetchWaitTime analog (stream-time; can exceed wall)\n",
      to_milliseconds(gpu_timeline.cross_stream_stall_microseconds));
  std::printf(
      "  GPU execution (measured): %8.3f ms  summed GPU time"
      " over %d/%zu nodes (%s%s)\n",
      to_milliseconds(gpu_timeline.measured_gpu_compute_microseconds),
      gpu_timeline.gpu_measured_node_count,
      gpu_timeline.nodes.size(),
      gpu_vocabulary::CONSOLE_TIMESTAMP_SOURCE,
      gpu_timeline.granularity == TimelineGranularity::per_stage
          ? gpu_vocabulary::CONSOLE_STAGE_NOTE
          : "");
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
  // A per-stage pass never timed nodes individually; zeros would lie.
  if (gpu_timeline.granularity != TimelineGranularity::per_node) {
    return;
  }
  std::printf("\n== nodes (timeline pass; µs since pass start) ==\n");
  std::printf(
      "  %4s %-6s %-20s %3s %4s %8s %8s %7s %7s %7s %7s %8s %7s %8s  %s\n",
      "#", "node", "op", "str", "join", "issued@", "gpu@", "cpu", "gpu-q",
      "dep", "sched", "compute", "result", "wall", "class");
  int dispatch_position = 0;
  for (const NodeTimeline& node_timeline : gpu_timeline.nodes) {
    const LatencyBreakdown& breakdown = node_timeline.breakdown;
    const NodeTimestamps& timestamps = node_timeline.timestamps;
    const std::string_view class_name =
        node_class_name(breakdown.classification);
    // "~" marks a residual (unmeasured) compute so it is never mistaken for
    // GPU-measured time.
    std::printf(
        "  %4d %-6s %-20.20s %3d %4s %8.1f %8s %7.1f %7.1f %7.1f %7.1f %7.1f%1s"
        " %7.1f %8.1f  %.*s\n",
        ++dispatch_position,
        node_timeline.node->label.c_str(),
        node_timeline.node->op_name.c_str(),
        node_timeline.node->stream_index,
        node_timeline.has_cross_stream_input ? "yes" : "",
        timestamps.issue_started_microseconds,
        format_gpu_start(timestamps).c_str(),
        breakdown.wait_on_cpu_microseconds,
        breakdown.wait_on_gpu_microseconds,
        breakdown.wait_on_dependency_microseconds,
        breakdown.scheduler_delay_microseconds,
        breakdown.compute_microseconds,
        breakdown.is_compute_measured ? "" : "~",
        breakdown.completion_latency_microseconds,
        breakdown.wall_microseconds,
        static_cast<int>(class_name.size()),
        class_name.data());
  }
  std::printf(
      "  compute = GPU command-buffer execution (GPUEndTime − GPUStartTime)"
      " minus in-buffer waits; \"~\" = host residual (no GPU window).\n"
      "  gpu-q = queued behind earlier work on the stream; sched = committed"
      " and unblocked, GPU not started yet (Spark scheduler delay).\n"
      "  result = GPU end → host noticed (Spark \"getting result\"), excluded"
      " from class. Granularity is one command buffer per node.\n");
}

void print_stage_table(const GpuTimeline& gpu_timeline) {
  if (gpu_timeline.granularity != TimelineGranularity::per_stage) {
    return;
  }
  std::printf("\n== stages (timeline pass; one async_eval per stage; µs) ==\n");
  std::printf("  %3s %-22s %3s %5s %8s %8s %7s %7s %7s %7s %8s %7s %8s  %s\n", "#",
              "stage", "str", "nodes", "issued@", "gpu@", "cpu", "gpu-q", "dep", "sched",
              "compute", "result", "wall", "class");
  int stage_task_number = 0;
  for (const StageTimeline& stage_timeline : gpu_timeline.stages) {
    const LatencyBreakdown& breakdown = stage_timeline.breakdown;
    const std::string_view class_name = node_class_name(breakdown.classification);
    std::printf(
        "  %3d %-22.22s %3d %5zu %8.1f %8s %7.1f %7.1f %7.1f %7.1f %7.1f%1s %7.1f %8.1f"
        "  %.*s\n",
        ++stage_task_number, stage_timeline.stage_name.c_str(), stage_timeline.stream_index,
        stage_timeline.node_positions.size(),
        stage_timeline.timestamps.issue_started_microseconds,
        format_gpu_start(stage_timeline.timestamps).c_str(),
        breakdown.wait_on_cpu_microseconds, breakdown.wait_on_gpu_microseconds,
        breakdown.wait_on_dependency_microseconds, breakdown.scheduler_delay_microseconds,
        breakdown.compute_microseconds, breakdown.is_compute_measured ? "" : "~",
        breakdown.completion_latency_microseconds, breakdown.wall_microseconds,
        static_cast<int>(class_name.size()), class_name.data());
  }
  std::printf(
      "  compute = first command buffer's GPU start → end of the stage's last buffer"
      " (marker-bounded); includes any GPU gap while the host still encoded the stage.\n"
      "  gpu-q = issued, but the stream was still busy with earlier stages (the host"
      " ran ahead of the GPU).\n");
}

} // namespace mlx_perf
