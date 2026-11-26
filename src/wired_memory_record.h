#pragma once

// Wired memory record: one concern only — express a run's wired memory
// samples and their analysis as JSON (described in docs/measurement-file.md),
// shared by the measurement file (`job.wired_memory`) and the procedure
// comparison record (`baseline.wired_memory`, `candidate.wired_memory`).
// Pure: no files, no system calls.

#include "json_value.h"
#include "wired_memory_sample.h"

#include <vector>

namespace mlx_perf {

// null when there are no samples. Times are microseconds since the first
// sample; a figure that needs the wire limit is null when it is unknown.
JsonValue build_wired_memory_record(const std::vector<WiredMemorySample>& wired_memory_samples);

} // namespace mlx_perf
