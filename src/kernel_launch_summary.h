#pragma once

// Kernel launch summary: one concern only — condense the kernel launches the
// serial pass recorded into what a reader compares: per kernel, how many
// launches, how many threadgroups each formed, and the lowest share of GPU
// cores any of them could use. Pure; the report sections format the result.

#include "telemetry.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlx_perf {

struct KernelLaunchSummary {
  // What a reader recognises the kernel by: a custom kernel's name, else the
  // MLX operation (two variants' "Softmax" nodes pair up this way).
  std::string kernel_key;
  int launch_count = 0;
  std::uint64_t fewest_threadgroups = 0;
  std::uint64_t most_threadgroups = 0;
  // The lowest gpu_core_coverage among the launches; negative when the core
  // count is unknown.
  double lowest_core_coverage = -1.0;
  bool has_launch_unable_to_fill_gpu = false;
};

std::string kernel_key_of(const GraphNode& graph_node);

// One summary per kernel key, in the order the kernels first ran. Nodes
// that launched nothing (views such as a reshape) are left out.
std::vector<KernelLaunchSummary> summarize_kernel_launches(const RunTelemetry& serial_telemetry);

// A node's launches in one short cell, e.g. "4 threadgroups · 4 of 16 cores"
// or "2 launches · 8–256 threadgroups · 8 of 16 cores"; "" when it launched
// nothing.
std::string describe_node_launches(const std::vector<KernelDispatch>& kernel_dispatches,
                                   int gpu_core_count);

} // namespace mlx_perf
