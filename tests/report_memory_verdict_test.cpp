// Component tests for report_memory_verdict: the judged memory table and the
// memory pressure note, from synthetic comparisons. No GPU.

#include "report_memory_verdict.h"
#include "test_harness.h"

#include <string>
#include <vector>

namespace {

using namespace mlx_perf;

bool contains(const std::string& page_html, const std::string& needle) {
  return page_html.find(needle) != std::string::npos;
}

std::vector<double> mebibytes(std::initializer_list<double> mebibyte_counts) {
  std::vector<double> byte_counts;
  for (const double mebibyte_count : mebibyte_counts) byte_counts.push_back(mebibyte_count * 1048576.0);
  return byte_counts;
}

// --- positive ---

void should_judge_memory_in_bytes_with_less_and_more() {
  const std::vector<double> baseline_bytes = mebibytes({64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75});
  const std::vector<double> candidate_bytes = mebibytes({320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331});
  const PairedComparison larger = compare_paired_samples(baseline_bytes, candidate_bytes);
  const PairedComparison unchanged = compare_paired_samples(baseline_bytes, baseline_bytes);
  const std::string table_html = render_memory_verdict_table("chatty", "batched", {&larger, &unchanged, &unchanged});
  EXPECT_TRUE(contains(table_html, "Peak MLX working set"));
  EXPECT_TRUE(contains(table_html, "Rounds batched used less"));
  EXPECT_TRUE(contains(table_html, "more, beyond the spread"));
  EXPECT_TRUE(contains(table_html, "+256.0 MiB"));
  EXPECT_TRUE(contains(table_html, "inside the spread: not a result"));
}

void should_name_each_sides_crossings_and_compressions() {
  const std::string note_html = render_memory_pressure_note({"chatty", 2, 0, 12}, {"batched", 0, 3, 12});
  EXPECT_TRUE(contains(note_html, "class=\"note memory-pressure\""));
  EXPECT_TRUE(contains(note_html, "chatty crossed the wire limit in 2 of 12 rounds"));
  EXPECT_TRUE(contains(note_html, "batched crossed the wire limit in 0 of 12 rounds and saw the system compress memory in 3"));
}

// --- edge / boundary ---

void should_say_nothing_when_no_round_was_under_pressure() {
  EXPECT_TRUE(render_memory_pressure_note({"chatty", 0, 0, 12}, {"batched", 0, 0, 12}).empty());
}

void should_write_a_negative_amount_left_behind_with_a_minus_sign() {
  const PairedComparison freed_more =
      compare_paired_samples(mebibytes({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
                             mebibytes({-8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8}));
  const std::string table_html = render_memory_verdict_table("a", "b", {&freed_more, &freed_more, &freed_more});
  EXPECT_TRUE(contains(table_html, "−8.0 MiB"));
  EXPECT_TRUE(contains(table_html, "less, beyond the spread"));
}

// --- negative ---

void should_escape_side_names() {
  const std::string note_html = render_memory_pressure_note({"<b>a</b>", 1, 0, 12}, {"b", 0, 0, 12});
  EXPECT_TRUE(!contains(note_html, "<b>a</b>"));
  EXPECT_TRUE(contains(note_html, "&lt;b&gt;a&lt;/b&gt;"));
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_judge_memory_in_bytes_with_less_and_more),
      TEST_CASE(should_name_each_sides_crossings_and_compressions),
      TEST_CASE(should_say_nothing_when_no_round_was_under_pressure),
      TEST_CASE(should_write_a_negative_amount_left_behind_with_a_minus_sign),
      TEST_CASE(should_escape_side_names),
  });
}
