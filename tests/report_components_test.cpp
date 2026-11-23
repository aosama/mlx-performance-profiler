// Unit tests for report_components: the stacked bar, legend and badge.

#include "report_components.h"
#include "test_harness.h"

#include <regex>
#include <string>

namespace {

using namespace mlx_plan;

std::size_t count_occurrences(const std::string& text, const std::string& needle) {
  std::size_t occurrence_count = 0;
  for (std::size_t found_at = text.find(needle); found_at != std::string::npos;
       found_at = text.find(needle, found_at + needle.size())) {
    ++occurrence_count;
  }
  return occurrence_count;
}

double sum_segment_widths(const std::string& bar_html) {
  const std::regex width_pattern("width:([0-9.eE+-]+)%");
  double summed_percent = 0.0;
  for (std::sregex_iterator match(bar_html.begin(), bar_html.end(), width_pattern);
       match != std::sregex_iterator(); ++match) {
    summed_percent += std::stod((*match)[1].str());
  }
  return summed_percent;
}

// --- positive ---

void should_size_bar_segments_to_sum_to_one_hundred_percent() {
  LatencyBreakdown breakdown;
  breakdown.wait_on_cpu_microseconds = 10;
  breakdown.wait_on_dependency_microseconds = 30;
  breakdown.compute_microseconds = 60;
  breakdown.wall_microseconds = 100;
  const std::string bar_html = render_stacked_bar(breakdown);
  EXPECT_NEAR(sum_segment_widths(bar_html), 100.0, 1e-6);
  EXPECT_TRUE(count_occurrences(bar_html, "<span") == 3);
}

void should_list_all_six_slices_in_the_legend() {
  EXPECT_TRUE(count_occurrences(render_slice_legend(), "class=\"swatch") == 6);
}

void should_color_class_badge_like_its_slice() {
  const std::string badge_html = render_class_badge(NodeClass::wait_on_dependency);
  EXPECT_TRUE(badge_html.find("slice-stalled") != std::string::npos);
  EXPECT_TRUE(badge_html.find("wait_on_dependency") != std::string::npos);
}

// --- edge / boundary ---

void should_render_empty_bar_for_zero_wall() {
  const std::string bar_html = render_stacked_bar(LatencyBreakdown{});
  EXPECT_TRUE(bar_html.find("<span") == std::string::npos);
  EXPECT_TRUE(bar_html.find("stacked-bar") != std::string::npos);
}

void should_omit_zero_length_slices_from_the_bar() {
  LatencyBreakdown breakdown;
  breakdown.compute_microseconds = 5;
  breakdown.wall_microseconds = 5;
  EXPECT_TRUE(count_occurrences(render_stacked_bar(breakdown), "<span") == 1);
}

// --- negative ---

void should_not_draw_negative_slices_from_corrupt_input() {
  LatencyBreakdown breakdown;
  breakdown.wait_on_gpu_microseconds = -4;
  breakdown.compute_microseconds = 8;
  breakdown.wall_microseconds = 8;
  const std::string bar_html = render_stacked_bar(breakdown);
  EXPECT_TRUE(bar_html.find("width:-") == std::string::npos);
  EXPECT_TRUE(count_occurrences(bar_html, "<span") == 1);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_size_bar_segments_to_sum_to_one_hundred_percent),
      TEST_CASE(should_list_all_six_slices_in_the_legend),
      TEST_CASE(should_color_class_badge_like_its_slice),
      TEST_CASE(should_render_empty_bar_for_zero_wall),
      TEST_CASE(should_omit_zero_length_slices_from_the_bar),
      TEST_CASE(should_not_draw_negative_slices_from_corrupt_input),
  });
}
