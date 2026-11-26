// Component tests for report_wired_memory_section: the wired memory lane,
// from synthetic samples. No GPU.

#include "report_wired_memory_section.h"
#include "test_harness.h"
#include "wired_memory_fixture.h"

#include <string>

namespace {

using namespace mlx_perf;
using wired_memory_fixture::GIB;
using wired_memory_fixture::sample_at;

const WiredMemorySectionText SECTION_TEXT{"wired-memory", "Wired memory", "Sampled in a test."};

bool contains(const std::string& section_html, const std::string& needle) {
  return section_html.find(needle) != std::string::npos;
}

// --- positive ---

void should_draw_the_three_series_the_release_and_its_lag() {
  const std::string section_html = render_wired_memory_section(SECTION_TEXT, wired_memory_fixture::release_then_unwire());
  EXPECT_TRUE(contains(section_html, "<section id=\"wired-memory\">"));
  EXPECT_TRUE(contains(section_html, "class=\"system-wired\" points="));
  EXPECT_TRUE(contains(section_html, "class=\"mlx-held\" points="));
  EXPECT_TRUE(contains(section_html, "class=\"outside-mlx\" points="));
  EXPECT_TRUE(contains(section_html, "class=\"release\""));
  EXPECT_TRUE(contains(section_html, "un-wired after 30.00 ms"));
  EXPECT_TRUE(contains(section_html, "class=\"pending-unwire\" r="));
  EXPECT_TRUE(contains(section_html, "Sampled in a test."));
}

void should_summarize_the_limit_peaks_and_release() {
  const std::string section_html = render_wired_memory_section(SECTION_TEXT, wired_memory_fixture::release_then_unwire());
  EXPECT_TRUE(contains(section_html, ">40.00 GiB<"));
  EXPECT_TRUE(contains(section_html, ">4.00 GiB<"));
  EXPECT_TRUE(contains(section_html, "1 release; un-wired after 30.00 ms (median)"));
  EXPECT_TRUE(contains(section_html, "3 (by up to 1.00 GiB)"));
}

// --- edge / boundary ---

void should_render_nothing_without_samples() {
  EXPECT_TRUE(render_wired_memory_section(SECTION_TEXT, {}).empty());
}

void should_draw_the_limit_line_only_when_it_sits_near_the_data() {
  EXPECT_TRUE(!contains(render_wired_memory_section(SECTION_TEXT, {sample_at(0, 3 * GIB, 0)}), "class=\"limit\""));
  EXPECT_TRUE(contains(render_wired_memory_section(SECTION_TEXT, {sample_at(0, 30 * GIB, 0)}), "class=\"limit\""));
}

void should_mark_evaluation_boundaries() {
  WiredMemorySample boundary_sample = sample_at(1, 3 * GIB, 0);
  boundary_sample.sample_point = WiredSamplePoint::evaluation_boundary;
  const std::string section_html = render_wired_memory_section(SECTION_TEXT, {sample_at(0, 3 * GIB, 0), boundary_sample});
  EXPECT_TRUE(contains(section_html, "<line class=\"boundary\""));
}

// --- negative ---

void should_escape_the_heading_and_say_when_the_limit_is_unknown() {
  const std::string section_html = render_wired_memory_section(
      {"wired-memory", "Wired memory: <b>", "note"}, {sample_at(0, 3 * GIB, 0, 0, 0)});
  EXPECT_TRUE(contains(section_html, "Wired memory: &lt;b&gt;"));
  EXPECT_TRUE(contains(section_html, ">unknown<"));
  EXPECT_TRUE(contains(section_html, "no release of 64.0 MiB or more"));
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_draw_the_three_series_the_release_and_its_lag),
      TEST_CASE(should_summarize_the_limit_peaks_and_release),
      TEST_CASE(should_render_nothing_without_samples),
      TEST_CASE(should_draw_the_limit_line_only_when_it_sits_near_the_data),
      TEST_CASE(should_mark_evaluation_boundaries),
      TEST_CASE(should_escape_the_heading_and_say_when_the_limit_is_unknown),
  });
}
