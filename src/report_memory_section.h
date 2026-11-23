#pragma once

// Memory views: one concern only — Spark's storage/executor memory angle:
// which nodes added memory in a run, and the process's memory over time.

#include "live_view_store.h"
#include "measurement_session.h"

#include <string>
#include <vector>

namespace mlx_plan {

// Per-run: peak, and nodes ranked by active memory added (serial pass).
std::string render_memory_section(const VariantMeasurement& variant_measurement);

// Live: active / cache / peak bytes over the sampled window, as inline SVG.
std::string render_memory_timeline(const std::vector<MemorySample>& memory_samples);

} // namespace mlx_plan
