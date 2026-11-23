#pragma once

// Report stage table section: one concern only — Spark's per-task table at
// STAGE granularity: one sortable row per stage task (a stage on one
// stream), with its slices, the op mix that ran inside it, and the serial
// pass's per-node numbers summed over its members.

#include "report_inputs.h"

#include <string>
#include <vector>

namespace mlx_plan {

std::string render_stage_table_section(const ReportInputs& report_inputs);

// "QuantizedMatmul ×9, RMSNorm ×3, …": the most frequent ops among
// `member_positions` (positions in gpu_timeline.nodes), most frequent first,
// ties by name; at most `shown_op_count` entries.
std::string summarize_op_mix(const GpuTimeline& gpu_timeline,
                             const std::vector<int>& member_positions,
                             int shown_op_count);

} // namespace mlx_plan
