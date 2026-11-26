// Unit tests for text_format: unit boundaries and escaping.

#include "test_harness.h"
#include "text_format.h"

namespace {

using namespace mlx_perf;

// --- positive ---

void should_format_sub_millisecond_durations_in_microseconds() {
  EXPECT_TRUE(format_microseconds(412.34) == "412.3 µs");
}

void should_format_durations_from_one_millisecond_in_milliseconds() {
  EXPECT_TRUE(format_microseconds(1240.0) == "1.24 ms");
}

void should_format_byte_sizes_with_binary_units() {
  EXPECT_TRUE(format_bytes(512) == "512 B");
  EXPECT_TRUE(format_bytes(3584) == "3.5 KiB");
  EXPECT_TRUE(format_bytes(1258291) == "1.2 MiB");
}

void should_escape_every_html_special_character() {
  EXPECT_TRUE(escape_html_text("<a href=\"x\">&'</a>") ==
              "&lt;a href=&quot;x&quot;&gt;&amp;&#39;&lt;/a&gt;");
}

void should_turn_real_newlines_into_dot_line_breaks() {
  EXPECT_TRUE(escape_dot_text("A\nMatMul") == "A\\nMatMul");
  EXPECT_TRUE(escape_dot_text("say \"hi\"\\") == "say \\\"hi\\\"\\\\");
}

void should_percent_encode_a_file_name_for_a_link() {
  EXPECT_TRUE(encode_url_path_segment("bundle_v2#fast?.html") == "bundle_v2%23fast%3F.html");
  EXPECT_TRUE(encode_url_path_segment("50% faster") == "50%25%20faster");
  EXPECT_TRUE(encode_url_path_segment("plain-name_1.html") == "plain-name_1.html");
}

void should_compare_names_ignoring_ascii_case() {
  EXPECT_TRUE(are_equal_ignoring_case("Stock", "stock"));
  EXPECT_TRUE(!are_equal_ignoring_case("stock", "stocks"));
  EXPECT_TRUE(!are_equal_ignoring_case("stock", "custom"));
}

// --- edge / boundary ---

void should_switch_units_exactly_at_one_millisecond() {
  EXPECT_TRUE(format_microseconds(999.9) == "999.9 µs");
  EXPECT_TRUE(format_microseconds(1000.0) == "1.00 ms");
}

void should_switch_units_exactly_at_one_kibibyte() {
  EXPECT_TRUE(format_bytes(1023) == "1023 B");
  EXPECT_TRUE(format_bytes(1024) == "1.0 KiB");
}

void should_format_zero_values() {
  EXPECT_TRUE(format_microseconds(0.0) == "0.0 µs");
  EXPECT_TRUE(format_bytes(0) == "0 B");
  EXPECT_TRUE(escape_html_text("").empty());
}

// --- negative ---

void should_encode_and_compare_empty_names() {
  EXPECT_TRUE(encode_url_path_segment("").empty());
  EXPECT_TRUE(are_equal_ignoring_case("", ""));
}

void should_encode_every_byte_of_non_ascii_text() {
  EXPECT_TRUE(encode_url_path_segment("\xce\xb1") == "%CE%B1");  // "α"
}

void should_keep_the_sign_of_negative_durations_rather_than_hide_them() {
  // A negative duration is a measurement bug upstream; formatting must show
  // it plainly, not clamp it into a plausible-looking number.
  EXPECT_TRUE(format_microseconds(-5.0) == "-5.0 µs");
}

void should_neutralize_script_injection_in_html_text() {
  const std::string escaped = escape_html_text("<script>alert(1)</script>");
  EXPECT_TRUE(escaped.find('<') == std::string::npos);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_format_sub_millisecond_durations_in_microseconds),
      TEST_CASE(should_format_durations_from_one_millisecond_in_milliseconds),
      TEST_CASE(should_format_byte_sizes_with_binary_units),
      TEST_CASE(should_escape_every_html_special_character),
      TEST_CASE(should_turn_real_newlines_into_dot_line_breaks),
      TEST_CASE(should_percent_encode_a_file_name_for_a_link),
      TEST_CASE(should_compare_names_ignoring_ascii_case),
      TEST_CASE(should_switch_units_exactly_at_one_millisecond),
      TEST_CASE(should_switch_units_exactly_at_one_kibibyte),
      TEST_CASE(should_format_zero_values),
      TEST_CASE(should_encode_and_compare_empty_names),
      TEST_CASE(should_encode_every_byte_of_non_ascii_text),
      TEST_CASE(should_keep_the_sign_of_negative_durations_rather_than_hide_them),
      TEST_CASE(should_neutralize_script_injection_in_html_text),
  });
}
