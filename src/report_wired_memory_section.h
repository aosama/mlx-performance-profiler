#pragma once

// Report wired memory section: one concern only — the "Wired memory" lane of
// a report or procedure comparison page: the system's wired total, MLX's own
// holding and the part wired outside MLX over the run, against the wire
// limit, with the evaluation boundaries, MLX's releases and how long macOS
// took to un-wire them, and the samples whose "outside MLX" that delay
// inflated (#45). Drawn as inline SVG, so the page stays one offline file.

#include "wired_memory_sample.h"

#include <string>
#include <vector>

namespace mlx_perf {

struct WiredMemorySectionText {
  std::string section_id;   // the HTML id, e.g. "wired-memory-baseline"
  std::string heading;      // e.g. "Wired memory: chatty"
  std::string source_note;  // where the samples come from, one sentence
};

// Empty when there are no samples.
std::string render_wired_memory_section(const WiredMemorySectionText& section_text,
                                        const std::vector<WiredMemorySample>& wired_memory_samples);

} // namespace mlx_perf
