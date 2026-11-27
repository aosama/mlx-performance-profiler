#pragma once

// Work record: one concern only — express the machine's ceilings and a timed
// unit's work and judgement as JSON (described in docs/measurement-file.md):
// `job.machine_ceilings`, and `data_and_arithmetic` on every node and stage.
// Pure: no files, no GPU.

#include "json_value.h"
#include "machine_ceilings.h"
#include "node_work.h"
#include "work_limits.h"

namespace mlx_perf {

// null when the ceilings were not measured.
JsonValue build_machine_ceilings_record(const MachineCeilings& machine_ceilings);

// null when the unit moved nothing (a view, which launches no kernel).
// Figures that need what is missing (ceilings, a GPU window, counted
// arithmetic, resolved expert indices) are null, never 0.
JsonValue build_work_record(const NodeWork& unit_work, const WorkJudgement& work_judgement);

} // namespace mlx_perf
