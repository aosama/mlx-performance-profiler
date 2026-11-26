// Unit and component tests for graph_view_section and the embedded graph
// view assets: escaping a document into a script element, element ids, and
// what a page needs to mount a view. No GPU.

#include "graph_view_assets.h"
#include "graph_view_section.h"
#include "test_harness.h"

#include <string>

namespace {

using namespace mlx_perf;

std::size_t count_occurrences(const std::string& text, const std::string& needle) {
  std::size_t occurrence_count = 0;
  for (std::size_t found_at = text.find(needle); found_at != std::string::npos;
       found_at = text.find(needle, found_at + needle.size())) {
    ++occurrence_count;
  }
  return occurrence_count;
}

// --- positive ---

void should_embed_the_document_beside_a_view_that_names_it() {
  const std::string view_html = render_graph_view("eager", "{\"nodes\": []}");
  EXPECT_TRUE(view_html.find("class=\"graph-view\" data-graph-document=\"graph-document-eager\"") !=
              std::string::npos);
  EXPECT_TRUE(view_html.find("<script type=\"application/json\" id=\"graph-document-eager\">"
                             "{\"nodes\": []}</script>") != std::string::npos);
}

void should_embed_every_library_and_viewer_script_in_load_order() {
  const std::string scripts_html = render_graph_view_scripts();
  EXPECT_TRUE(count_occurrences(scripts_html, "<script>") == 6);
  const std::size_t cytoscape_at = scripts_html.find("Cytoscape Consortium");
  const std::size_t dagre_layout_at = scripts_html.find("cytoscape-dagre");
  const std::size_t viewer_at = scripts_html.find("graphView.mountAll");
  EXPECT_TRUE(cytoscape_at != std::string::npos && dagre_layout_at != std::string::npos);
  EXPECT_TRUE(cytoscape_at < dagre_layout_at && dagre_layout_at < viewer_at);
}

void should_embed_the_stylesheet_from_its_source_file() {
  EXPECT_TRUE(render_graph_view_stylesheet().find(".graph-view") != std::string::npos);
}

void should_reduce_a_view_name_to_id_safe_characters() {
  EXPECT_TRUE(to_element_id_fragment("eager v2") == "eager-v2");
  EXPECT_TRUE(to_element_id_fragment("custom_kernel-1") == "custom_kernel-1");
}

// --- edge / boundary ---

void should_name_an_unnamed_view() {
  EXPECT_TRUE(to_element_id_fragment("") == "view");
}

void should_leave_json_without_angle_brackets_unchanged() {
  EXPECT_TRUE(escape_json_for_script_element("") == "");
  EXPECT_TRUE(escape_json_for_script_element("{\"a\": \"x > y / z\"}") ==
              "{\"a\": \"x > y / z\"}");
}

void should_escape_an_angle_bracket_at_either_end() {
  EXPECT_TRUE(escape_json_for_script_element("<") == "\\u003c");
  EXPECT_TRUE(escape_json_for_script_element("\"a<") == "\"a\\u003c");
}

// --- negative ---

void should_stop_a_kernel_name_from_closing_the_script_element() {
  const std::string view_html =
      render_graph_view("v", "{\"kernel_name\": \"</script><script>alert(1)</script>\"}");
  // The only closing tag is the element's own.
  EXPECT_TRUE(count_occurrences(view_html, "</script>") == 1);
  EXPECT_TRUE(view_html.find("\\u003c/script>\\u003cscript>alert(1)") != std::string::npos);
}

// "<!--" then "<script" would put the parser in a state where the element's
// own "</script>" no longer closes it, swallowing the rest of the page.
void should_stop_a_comment_opener_from_swallowing_the_page() {
  const std::string view_html =
      render_graph_view("v", "{\"op_name\": \"<!--<script>\"}");
  EXPECT_TRUE(view_html.find("<!--") == std::string::npos);
  EXPECT_TRUE(count_occurrences(view_html, "<script") == 1);
}

void should_strip_markup_from_a_view_name() {
  const std::string view_html = render_graph_view("\"><img src=x>", "{}");
  EXPECT_TRUE(view_html.find("<img") == std::string::npos);
  EXPECT_TRUE(view_html.find("graph-document----img-src-x-") != std::string::npos);
}

// A library update that ships "</script" inside its code would end the
// page's script element mid-file and silently break every report.
void should_embed_no_script_that_would_end_its_own_element() {
  for (const std::string_view embedded_script :
       {graph_view_cytoscape_script(), graph_view_dagre_layout_script(),
        graph_view_model_script(), graph_view_style_script(),
        graph_view_panels_script(), graph_view_viewer_script()}) {
    EXPECT_TRUE(!embedded_script.empty());
    EXPECT_TRUE(embedded_script.find("</script") == std::string_view::npos);
    EXPECT_TRUE(embedded_script.find("</SCRIPT") == std::string_view::npos);
  }
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_embed_the_document_beside_a_view_that_names_it),
      TEST_CASE(should_embed_every_library_and_viewer_script_in_load_order),
      TEST_CASE(should_embed_the_stylesheet_from_its_source_file),
      TEST_CASE(should_reduce_a_view_name_to_id_safe_characters),
      TEST_CASE(should_name_an_unnamed_view),
      TEST_CASE(should_leave_json_without_angle_brackets_unchanged),
      TEST_CASE(should_escape_an_angle_bracket_at_either_end),
      TEST_CASE(should_stop_a_kernel_name_from_closing_the_script_element),
      TEST_CASE(should_stop_a_comment_opener_from_swallowing_the_page),
      TEST_CASE(should_strip_markup_from_a_view_name),
      TEST_CASE(should_embed_no_script_that_would_end_its_own_element),
  });
}
