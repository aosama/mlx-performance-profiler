#pragma once

// Graph view section: one concern only — place the interactive graph view in
// a page: the embedded libraries and viewer (once per page), and one view per
// measurement document, with the document embedded beside it as JSON. The
// viewer (src/graph_view/) reads the same measurement document the profiler
// writes to disk, so the picture and the data file cannot disagree.

#include <string>

namespace mlx_perf {

// The graph view's stylesheet, for the page's <head>.
std::string render_graph_view_stylesheet();

// The libraries and the viewer, as <script> elements. Include once per page,
// after every view: the viewer mounts each view it finds when it runs.
std::string render_graph_view_scripts();

// One view of `measurement_document_json` (a measurement document, schema
// version 1). `view_name` becomes part of the element ids, reduced to
// letters, digits, '-' and '_'; callers give each view on a page a distinct
// name.
std::string render_graph_view(const std::string& view_name,
                              const std::string& measurement_document_json);

// JSON text made safe inside <script type="application/json">: every '<'
// becomes "<", which JSON reads back as the same string, so no op or
// kernel name can close the script element early or change how the HTML
// parser reads it.
std::string escape_json_for_script_element(const std::string& json_text);

// "eager v2" → "eager-v2": the characters HTML ids and CSS selectors accept
// without escaping. Empty input becomes "view".
std::string to_element_id_fragment(const std::string& view_name);

} // namespace mlx_perf
