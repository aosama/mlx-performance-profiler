#pragma once

// Live view rendering: one concern only — turn the latest measured variants
// into the four tab fragments (Job / Stages / Tasks / Memory) by reusing the
// static report sections. Runs on the main thread, right after measuring.

#include "live_view_store.h"
#include "measurement_session.h"

#include <string>
#include <vector>

namespace mlx_plan {

// `measured_variants` must be non-empty; with two or more, the Job tab leads
// with the first-vs-second comparison table.
RenderedViews render_live_views(
    const std::string& job_title,
    const std::vector<const VariantMeasurement*>& measured_variants);

} // namespace mlx_plan
