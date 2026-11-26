#pragma once

// SVG rendering: one concern only — lay out DOT through Graphviz's `dot`,
// writing the .dot artifact as a side effect the report wants to keep anyway.

#include <string>

namespace mlx_perf {

// Writes `dot_source` to `<output_directory>/<plan_file_base>.dot`, runs
// `dot -Tsvg` over it, and returns the rendered SVG text. Returns an empty
// string when Graphviz is absent; the report writer then shows the raw DOT.
std::string render_plan_svg(
    const std::string& dot_source,
    const std::string& output_directory,
    const std::string& plan_file_base);

} // namespace mlx_perf
