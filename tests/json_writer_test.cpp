// Unit tests for json_writer: every document it writes must parse back to
// the same tree, including the values JSON cannot represent directly.

#include "json_value.h"
#include "json_writer.h"
#include "test_harness.h"

#include <cmath>
#include <limits>
#include <string>

namespace {

using mlx_perf::JsonValue;
using mlx_perf::parse_json;
using mlx_perf::write_json;

JsonValue round_trip(const JsonValue& document) { return parse_json(write_json(document)); }

// --- positive ---

void should_round_trip_nested_objects_arrays_and_scalars() {
  const JsonValue document(JsonValue::Object{
      {"node_count", JsonValue(19.0)},
      {"is_measured", JsonValue(true)},
      {"kernel_name", JsonValue()},
      {"shape", JsonValue(JsonValue::Array{JsonValue(128.0), JsonValue(64.0)})},
      {"job", JsonValue(JsonValue::Object{{"wall_us", JsonValue(1234.5)}})}});
  const JsonValue parsed = round_trip(document);
  EXPECT_TRUE(parsed.at("node_count").as_number() == 19.0);
  EXPECT_TRUE(parsed.at("is_measured").as_bool());
  EXPECT_TRUE(parsed.at("kernel_name").is_null());
  EXPECT_TRUE(parsed.at("shape").as_array().size() == 2);
  EXPECT_TRUE(parsed.at("shape").as_array()[1].as_number() == 64.0);
  EXPECT_TRUE(parsed.at("job").at("wall_us").as_number() == 1234.5);
}

void should_write_numbers_in_their_shortest_exact_form() {
  EXPECT_TRUE(write_json(JsonValue(3.0)) == "3\n");
  EXPECT_TRUE(write_json(JsonValue(12.5)) == "12.5\n");
  EXPECT_TRUE(write_json(JsonValue(0.1)) == "0.1\n");
  const double awkward_value = 1.0 / 3.0;
  EXPECT_TRUE(round_trip(JsonValue(awkward_value)).as_number() == awkward_value);
}

void should_write_members_in_key_order() {
  const std::string json_text = write_json(JsonValue(JsonValue::Object{
      {"zeta", JsonValue(1.0)}, {"alpha", JsonValue(2.0)}}));
  EXPECT_TRUE(json_text.find("\"alpha\"") < json_text.find("\"zeta\""));
}

// --- edge / boundary ---

void should_write_empty_containers_compactly() {
  EXPECT_TRUE(write_json(JsonValue(JsonValue::Object{})) == "{}\n");
  EXPECT_TRUE(write_json(JsonValue(JsonValue::Array{})) == "[]\n");
  EXPECT_TRUE(round_trip(JsonValue(JsonValue::Array{})).as_array().empty());
}

void should_escape_quotes_backslashes_and_control_characters() {
  const std::string awkward_text = "say \"hi\"\\ \n\t\x01 end";
  EXPECT_TRUE(round_trip(JsonValue(awkward_text)).as_string() == awkward_text);
  EXPECT_TRUE(write_json(JsonValue(std::string("\x01"))) == "\"\\u0001\"\n");
}

void should_pass_utf8_text_through_unchanged() {
  const std::string kernel_name = "rms_norm_\xce\xb1";  // "rms_norm_α"
  EXPECT_TRUE(round_trip(JsonValue(kernel_name)).as_string() == kernel_name);
}

void should_write_large_whole_numbers_as_plain_digits() {
  // Byte counts and shapes must read back as integers in strict readers.
  EXPECT_TRUE(write_json(JsonValue(1000000.0)) == "1000000\n");
  EXPECT_TRUE(write_json(JsonValue(1e12)) == "1000000000000\n");
  EXPECT_TRUE(write_json(JsonValue(9007199254740992.0)) == "9007199254740992\n");
  EXPECT_TRUE(write_json(JsonValue(-4096.0)) == "-4096\n");
}

void should_keep_the_shortest_form_beyond_exact_integers() {
  // Past 2^53 a double is no longer an exact integer: shortest form it is.
  EXPECT_TRUE(round_trip(JsonValue(1e300)).as_number() == 1e300);
  EXPECT_TRUE(write_json(JsonValue(0.000001)) == "1e-06\n");
}

void should_write_zero_and_negative_numbers() {
  EXPECT_TRUE(write_json(JsonValue(0.0)) == "0\n");
  EXPECT_TRUE(round_trip(JsonValue(-42.25)).as_number() == -42.25);
}

// --- negative: values JSON cannot represent ---

void should_write_non_finite_numbers_as_null() {
  EXPECT_TRUE(write_json(JsonValue(std::numeric_limits<double>::quiet_NaN())) == "null\n");
  EXPECT_TRUE(write_json(JsonValue(std::numeric_limits<double>::infinity())) == "null\n");
  EXPECT_TRUE(write_json(JsonValue(-std::numeric_limits<double>::infinity())) == "null\n");
  EXPECT_TRUE(round_trip(JsonValue(std::nan(""))).is_null());
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_round_trip_nested_objects_arrays_and_scalars),
      TEST_CASE(should_write_numbers_in_their_shortest_exact_form),
      TEST_CASE(should_write_members_in_key_order),
      TEST_CASE(should_write_empty_containers_compactly),
      TEST_CASE(should_escape_quotes_backslashes_and_control_characters),
      TEST_CASE(should_pass_utf8_text_through_unchanged),
      TEST_CASE(should_write_large_whole_numbers_as_plain_digits),
      TEST_CASE(should_keep_the_shortest_form_beyond_exact_integers),
      TEST_CASE(should_write_zero_and_negative_numbers),
      TEST_CASE(should_write_non_finite_numbers_as_null),
  });
}
