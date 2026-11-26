#include "report_memory_verdict.h"

#include "report_verdict_section.h"
#include "text_format.h"

#include <sstream>

namespace mlx_perf {

// Said beside the timing verdict, because a round that crossed the wire
// limit or made the system compress memory is slow for a reason outside the
// code, and a timing difference built on such rounds is not the code's.
std::string render_memory_pressure_note(const MemoryPressureCounts& baseline_pressure,
                                        const MemoryPressureCounts& candidate_pressure) {
  if (baseline_pressure.rounds_crossing_wire_limit + baseline_pressure.rounds_with_compression +
          candidate_pressure.rounds_crossing_wire_limit + candidate_pressure.rounds_with_compression ==
      0) {
    return "";
  }
  std::ostringstream note_html;
  note_html << "<p class=\"note memory-pressure\">⚠ Memory pressure during the rounds.";
  for (const MemoryPressureCounts* side_pressure : {&baseline_pressure, &candidate_pressure}) {
    note_html << " " << escape_html_text(side_pressure->side_name) << " crossed the wire limit in "
              << side_pressure->rounds_crossing_wire_limit << " of " << side_pressure->round_count
              << " rounds and saw the system compress memory in " << side_pressure->rounds_with_compression << ".";
  }
  note_html << " Such rounds are slow for a reason outside the code, so read the timing verdict with that in"
            << " mind; compression may come from any program on the Mac.</p>";
  return note_html.str();
}

std::string render_memory_verdict_table(const std::string& baseline_name, const std::string& candidate_name,
                                        const MemoryComparisons& memory_comparisons) {
  std::ostringstream table_html;
  table_html << render_verdict_table(
                    baseline_name, candidate_name,
                    {{"Peak MLX working set (peak active less active at the start)",
                      memory_comparisons.peak_mlx_active_added},
                     {"Left behind (MLX active + cache, end less start)", memory_comparisons.mlx_held_change},
                     {"System wired at the run's end", memory_comparisons.system_wired_at_end}},
                    MetricUnit::bytes)
             << "<p class=\"note\">Judged like the times, on the same rounds, from figures read just before and"
             << " just after each run, never during it, so the timing pays nothing for them. The peak working set"
             << " is MLX's own peak counter less the active memory as the run began, because MLX's counters are"
             << " process-wide. Wired memory is read only at the run's end here; its true peak is in the wired"
             << " memory lanes (each variant's own report, or each procedure's observed run).</p>";
  return table_html.str();
}

} // namespace mlx_perf
