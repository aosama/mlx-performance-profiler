#pragma once

// Report launch comparison section: one concern only — the "Kernel launches
// side by side" table of the comparison and procedure comparison pages. For
// each kernel either side ran, matched by kernel name (graph jobs) or Metal
// function (procedures), it shows each side's launches, the threadgroups they
// formed and how much of the GPU they could use, and flags the side that
// cannot fill the GPU. This is the "why" behind a verdict when a kernel is
// slow because it leaves cores idle.

#include "kernel_launch_summary.h"
#include "telemetry.h"

#include <string>
#include <vector>

namespace mlx_perf {

// Empty when neither variant recorded a launch (nothing to compare).
std::string render_launch_comparison_section(const std::string& baseline_name,
                                             const RunTelemetry& baseline_serial_telemetry,
                                             const std::string& candidate_name,
                                             const RunTelemetry& candidate_serial_telemetry);

// The same table from launch summaries already grouped (by Metal function,
// for procedures). `key_heading` names what the first column holds.
std::string render_launch_comparison_section(const std::string& baseline_name,
                                             const std::vector<KernelLaunchSummary>& baseline_kernels,
                                             const std::string& candidate_name,
                                             const std::vector<KernelLaunchSummary>& candidate_kernels,
                                             int gpu_core_count, const std::string& key_heading);

} // namespace mlx_perf
