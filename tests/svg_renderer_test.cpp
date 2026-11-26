// Component tests for svg_renderer: Graphviz layout of a DOT file whose name
// carries caller-chosen text. Skips the layout checks when `dot` is absent
// (the report then falls back to raw DOT, which is not an error).

#include "svg_renderer.h"
#include "test_harness.h"

#include <fstream>
#include <string>

namespace {

using mlx_perf::render_plan_svg;

constexpr const char* SMALL_PLAN_DOT = "digraph plan { A -> B; }\n";

bool file_exists(const std::string& file_path) { return std::ifstream(file_path).good(); }

// --- positive ---

void should_lay_out_a_plan_as_svg_and_keep_the_dot_file() {
  const std::string plan_svg = render_plan_svg(SMALL_PLAN_DOT, ".", "svg_renderer_plan");
  EXPECT_TRUE(file_exists("svg_renderer_plan.dot"));
  if (!plan_svg.empty()) {
    EXPECT_TRUE(plan_svg.find("<svg") != std::string::npos);
  }
}

// --- edge / boundary: names with spaces and quotes still lay out ---

void should_lay_out_a_plan_whose_name_has_spaces_and_quotes() {
  const std::string plan_svg =
      render_plan_svg(SMALL_PLAN_DOT, ".", "svg_renderer_\"fused v2\"");
  EXPECT_TRUE(file_exists("svg_renderer_\"fused v2\".dot"));
  const bool has_graphviz = !render_plan_svg(SMALL_PLAN_DOT, ".", "svg_renderer_probe").empty();
  EXPECT_TRUE(!has_graphviz || !plan_svg.empty());
}

// --- negative: a name is never executed ---

void should_never_run_shell_syntax_in_a_name() {
  render_plan_svg(SMALL_PLAN_DOT, ".", "svg_renderer_$(touch svg_renderer_injected)`touch "
                                       "svg_renderer_injected`");
  EXPECT_TRUE(!file_exists("svg_renderer_injected"));
}

void should_return_no_svg_for_a_malformed_graph() {
  EXPECT_TRUE(render_plan_svg("this is not dot {", ".", "svg_renderer_malformed").empty());
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_lay_out_a_plan_as_svg_and_keep_the_dot_file),
      TEST_CASE(should_lay_out_a_plan_whose_name_has_spaces_and_quotes),
      TEST_CASE(should_never_run_shell_syntax_in_a_name),
      TEST_CASE(should_return_no_svg_for_a_malformed_graph),
  });
}
