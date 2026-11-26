#pragma once

// Graph view assets: one concern only — the JavaScript and CSS the
// interactive graph view needs, embedded at build time from
// third_party/ (the vendored Cytoscape.js and its dagre layout) and
// src/graph_view/ (this project's viewer). Defined in a source that
// cmake/mlx_perf_embed_text_assets.cmake generates.

#include <string_view>

namespace mlx_perf {

// Cytoscape.js (MIT), third_party/cytoscape/: graph model, canvas renderer,
// pan and zoom, PNG export.
std::string_view graph_view_cytoscape_script();
// cytoscape-dagre (MIT, bundles dagre), third_party/cytoscape-dagre/: the
// layered top-to-bottom layout a dataflow graph reads best in.
std::string_view graph_view_dagre_layout_script();
// The viewer, src/graph_view/, in load order: the model (a measurement
// document, schema version 1, into nodes and edges), the Cytoscape style,
// the hover and details panels, and the mounting and interaction.
std::string_view graph_view_model_script();
std::string_view graph_view_style_script();
std::string_view graph_view_panels_script();
std::string_view graph_view_viewer_script();
// src/graph_view/graph_view.css: the viewer's layout, on the report's
// colour tokens so light and dark themes follow the page.
std::string_view graph_view_stylesheet();

} // namespace mlx_perf
