#pragma once

// Report memory verdict: one concern only — the memory half of a comparison
// page (#46): the table judging each side's memory per run on the same
// rounds as its time, and the note, under the timing verdict, saying when
// rounds crossed the wire limit or saw the system compress memory. Shared by
// the graph comparison page and the procedure comparison page.

#include "sample_statistics.h"

#include <string>

namespace mlx_perf {

// One side's memory pressure over its judged rounds.
struct MemoryPressureCounts {
  std::string side_name;
  int rounds_crossing_wire_limit = 0;
  int rounds_with_compression = 0;
  int round_count = 0;
};

// Empty when no round of either side crossed the limit or compressed.
std::string render_memory_pressure_note(const MemoryPressureCounts& baseline_pressure,
                                        const MemoryPressureCounts& candidate_pressure);

struct MemoryComparisons {
  const PairedComparison* peak_mlx_active_added;
  const PairedComparison* mlx_held_change;
  const PairedComparison* system_wired_at_end;
};

// The judged memory table and its note, without a heading or section: the
// page decides where it sits.
std::string render_memory_verdict_table(const std::string& baseline_name, const std::string& candidate_name,
                                        const MemoryComparisons& memory_comparisons);

} // namespace mlx_perf
