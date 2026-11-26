#pragma once

// Report launch comparison section: one concern only — the comparison page's
// "Kernel launches side by side" table. For each kernel either variant ran,
// matched by name, it shows each variant's launches, the threadgroups they
// formed and how much of the GPU they could use, and flags the variant that
// cannot fill the GPU. This is the "why" behind a verdict when a kernel is
// slow because it leaves cores idle.

#include "telemetry.h"

#include <string>

namespace mlx_perf {

// Empty when neither variant recorded a launch (nothing to compare).
std::string render_launch_comparison_section(const std::string& baseline_name,
                                             const RunTelemetry& baseline_serial_telemetry,
                                             const std::string& candidate_name,
                                             const RunTelemetry& candidate_serial_telemetry);

} // namespace mlx_perf
