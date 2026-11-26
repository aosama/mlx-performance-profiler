// Component tests for svg_renderer: Graphviz layout of a DOT file whose name
// carries caller-chosen text. Skips the layout checks when `dot` is absent
// (the report then falls back to raw DOT, which is not an error).

#include "captured_console.h"
#include "console_output.h"
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

// With console output off, even dot's complaint about a bad graph is kept
// off the caller's stderr.
void should_keep_dots_errors_off_the_console_when_output_is_off() {
  mlx_perf::set_console_output_enabled(false);
  captured_console::CapturedConsole captured_console("svg_renderer_silenced");
  const std::string plan_svg = render_plan_svg("this is not dot {", ".", "svg_renderer_bad_quiet");
  const std::string stderr_text = captured_console.stderr_text();
  mlx_perf::set_console_output_enabled(true);
  EXPECT_TRUE(plan_svg.empty());
  EXPECT_TRUE(stderr_text.empty());
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
      TEST_CASE(should_keep_dots_errors_off_the_console_when_output_is_off),
      TEST_CASE(should_return_no_svg_for_a_malformed_graph),
  });
}
