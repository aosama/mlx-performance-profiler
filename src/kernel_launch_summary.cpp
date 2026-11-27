#include "kernel_launch_summary.h"

#include <algorithm>
#include <sstream>

namespace mlx_perf {
namespace {

void add_launch(KernelLaunchSummary& kernel_summary, const KernelDispatch& kernel_dispatch,
                int gpu_core_count) {
  const std::uint64_t threadgroup_count = count_threadgroups(kernel_dispatch);
  kernel_summary.fewest_threadgroups = kernel_summary.launch_count == 0
      ? threadgroup_count : std::min(kernel_summary.fewest_threadgroups, threadgroup_count);
  kernel_summary.most_threadgroups = std::max(kernel_summary.most_threadgroups, threadgroup_count);
  const double core_coverage = gpu_core_coverage(threadgroup_count, gpu_core_count);
  if (core_coverage >= 0.0) {
    kernel_summary.lowest_core_coverage = kernel_summary.lowest_core_coverage < 0.0
        ? core_coverage : std::min(kernel_summary.lowest_core_coverage, core_coverage);
  }
  kernel_summary.has_launch_unable_to_fill_gpu |=
      gpu_core_count > 0 && !can_fill_gpu(threadgroup_count, gpu_core_count);
  ++kernel_summary.launch_count;
}

std::string describe_threadgroup_range(std::uint64_t fewest, std::uint64_t most) {
  std::ostringstream range_text;
  range_text << fewest;
  if (most != fewest) {
    range_text << "–" << most;
  }
  range_text << (most == 1 ? " threadgroup" : " threadgroups");
  return range_text.str();
}

KernelLaunchSummary& find_or_add_summary(std::vector<KernelLaunchSummary>& kernel_summaries,
                                         const std::string& kernel_key) {
  for (KernelLaunchSummary& kernel_summary : kernel_summaries) {
    if (kernel_summary.kernel_key == kernel_key) {
      return kernel_summary;
    }
  }
  kernel_summaries.push_back(KernelLaunchSummary{kernel_key});
  return kernel_summaries.back();
}

} // namespace

std::string kernel_key_of(const GraphNode& graph_node) {
  return graph_node.kernel_name.empty() ? graph_node.op_name : graph_node.kernel_name;
}

std::vector<KernelLaunchSummary> summarize_launches_by_function(
    const std::vector<KernelDispatch>& kernel_dispatches, int gpu_core_count) {
  std::vector<KernelLaunchSummary> function_summaries;
  for (const KernelDispatch& kernel_dispatch : kernel_dispatches) {
    const std::string function_key = kernel_dispatch.kernel_function_name.empty()
        ? std::string("(unnamed pipeline)") : kernel_dispatch.kernel_function_name;
    add_launch(find_or_add_summary(function_summaries, function_key), kernel_dispatch, gpu_core_count);
  }
  return function_summaries;
}

std::vector<KernelLaunchSummary> summarize_kernel_launches(const RunTelemetry& serial_telemetry) {
  std::vector<KernelLaunchSummary> kernel_summaries;
  const int gpu_core_count = serial_telemetry.gpu_device_facts.gpu_core_count;
  for (const NodeTelemetry& node_telemetry : serial_telemetry.attributed_nodes) {
    if (node_telemetry.node == nullptr || node_telemetry.kernel_dispatches.empty()) {
      continue;
    }
    KernelLaunchSummary& kernel_summary = find_or_add_summary(kernel_summaries, kernel_key_of(*node_telemetry.node));
    for (const KernelDispatch& kernel_dispatch : node_telemetry.kernel_dispatches) {
      add_launch(kernel_summary, kernel_dispatch, gpu_core_count);
    }
  }
  return kernel_summaries;
}

std::string describe_node_launches(const std::vector<KernelDispatch>& kernel_dispatches,
                                   int gpu_core_count) {
  if (kernel_dispatches.empty()) {
    return std::string();
  }
  KernelLaunchSummary node_summary;
  for (const KernelDispatch& kernel_dispatch : kernel_dispatches) {
    add_launch(node_summary, kernel_dispatch, gpu_core_count);
  }
  std::ostringstream description;
  if (node_summary.launch_count > 1) {
    description << node_summary.launch_count << " launches · ";
  }
  description << describe_threadgroup_range(node_summary.fewest_threadgroups,
                                            node_summary.most_threadgroups);
  if (gpu_core_count > 0) {
    const std::uint64_t cores_with_work =
        std::min<std::uint64_t>(node_summary.fewest_threadgroups, static_cast<std::uint64_t>(gpu_core_count));
    description << " · " << cores_with_work << " of " << gpu_core_count << " cores";
  }
  return description.str();
}

} // namespace mlx_perf
