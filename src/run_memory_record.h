#pragma once

// Run memory record: one concern only — express one run's memory figures,
// and a set of runs' memory pressure, as JSON (described in
// docs/measurement-file.md). Shared by the graph comparison record and the
// procedure comparison record (#46). Pure: no files, no system calls.

#include "json_value.h"
#include "run_memory_meter.h"

#include <vector>

namespace mlx_perf {

// The kernel's pressure figures are null when it refused to read them, so a
// "did not cross" or "did not compress" is never claimed without evidence.
JsonValue build_run_memory_record(const RunMemoryFigures& run_memory_figures);

// {"rounds_crossing_wire_limit", "rounds_with_compression", "rounds": [...]}.
JsonValue build_rounds_memory_record(const std::vector<RunMemoryFigures>& round_memory_figures);

} // namespace mlx_perf
