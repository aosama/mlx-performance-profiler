#include "graph_view_section.h"

#include "graph_view_assets.h"

#include <cctype>

namespace mlx_perf {

// Every '<', not only "</": "<!--" followed by "<script" switches the HTML
// parser into a state where the element's own "</script>" no longer closes
// it. JSON allows '<' only inside strings, where "<" reads back the same.
std::string escape_json_for_script_element(const std::string& json_text) {
  std::string escaped_json;
  escaped_json.reserve(json_text.size());
  for (const char json_character : json_text) {
    if (json_character == '<') {
      escaped_json += "\\u003c";
    } else {
      escaped_json += json_character;
    }
  }
  return escaped_json;
}

std::string to_element_id_fragment(const std::string& view_name) {
  std::string id_fragment;
  for (const char name_character : view_name) {
    const bool is_kept = std::isalnum(static_cast<unsigned char>(name_character)) != 0 ||
        name_character == '-' || name_character == '_';
    id_fragment += is_kept ? name_character : '-';
  }
  return id_fragment.empty() ? "view" : id_fragment;
}

std::string render_graph_view_stylesheet() {
  return std::string(graph_view_stylesheet());
}

std::string render_graph_view_scripts() {
  // Separate elements, so a parse error in one cannot take the others down
  // and the browser reports it against the right file.
  std::string scripts_html;
  for (const std::string_view script_text :
       {graph_view_cytoscape_script(), graph_view_dagre_layout_script(),
        graph_view_model_script(), graph_view_style_script(),
        graph_view_panels_script(), graph_view_viewer_script()}) {
    scripts_html += "<script>";
    scripts_html += script_text;
    scripts_html += "</script>\n";
  }
  return scripts_html;
}

std::string render_graph_view(const std::string& view_name,
                              const std::string& measurement_document_json) {
  const std::string document_element_id =
      "graph-document-" + to_element_id_fragment(view_name);
  return "<div class=\"graph-view\" data-graph-document=\"" + document_element_id +
         "\"><p class=\"note\">The interactive graph needs JavaScript.</p></div>"
         "<script type=\"application/json\" id=\"" + document_element_id + "\">" +
         escape_json_for_script_element(measurement_document_json) + "</script>";
}

} // namespace mlx_perf
