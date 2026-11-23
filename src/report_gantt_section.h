#pragma once

// Report Gantt section: one concern only — the per-stream timeline (Spark's
// event timeline / task-assignment Gantt) as inline SVG, plus the
// per-stream summary table that reads alongside it.

#include "report_inputs.h"

#include <string>

namespace mlx_plan {

std::string render_gantt_section(const ReportInputs& report_inputs);

// Just the SVG chart. `scale_total_microseconds` sets the time axis length,
// so several timelines can share one scale for side-by-side comparison; it
// is never shorter than this timeline's own wall.
std::string render_gantt_svg(
    const GpuTimeline& gpu_timeline, double scale_total_microseconds);

} // namespace mlx_plan
