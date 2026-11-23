// Unit tests for json_value: the parser that reads model config and vocab.

#include "json_value.h"
#include "test_harness.h"

#include <stdexcept>
#include <string>

namespace {

using mlx_plan::JsonValue;
using mlx_plan::parse_json;

bool throws_runtime_error(const std::string& json_text) {
  try {
    parse_json(json_text);
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}

// --- positive ---

void should_parse_nested_objects_arrays_and_scalars() {
  const JsonValue document = parse_json(
      R"({"hidden_size": 1024, "eps": 1e-6, "tied": true, "none": null,
          "layers": ["linear_attention", "full_attention"],
          "rope": {"theta": 10000000}})");
  EXPECT_TRUE(document.at("hidden_size").as_number() == 1024);
  EXPECT_NEAR(document.at("eps").as_number(), 1e-6, 1e-18);
  EXPECT_TRUE(document.at("tied").as_bool());
  EXPECT_TRUE(document.at("none").is_null());
  EXPECT_TRUE(document.at("layers").as_array()[1].as_string() == "full_attention");
  EXPECT_TRUE(document.at("rope").at("theta").as_number() == 1e7);
}

void should_decode_string_escapes_including_surrogate_pairs() {
  const JsonValue document = parse_json(R"(["a\"b\\c\n", "ĠParis", "😀"])");
  EXPECT_TRUE(document.as_array()[0].as_string() == "a\"b\\c\n");
  EXPECT_TRUE(document.as_array()[1].as_string() == "\xC4\xA0Paris");  // Ġ in UTF-8
  EXPECT_TRUE(document.as_array()[2].as_string() == "\xF0\x9F\x98\x80");
}

// --- edge / boundary ---

void should_parse_empty_containers_and_surrounding_whitespace() {
  const JsonValue document = parse_json(" \n\t{ \"a\" : [ ] , \"b\" : { } } \n");
  EXPECT_TRUE(document.at("a").as_array().empty());
  EXPECT_TRUE(document.at("b").as_object().empty());
  EXPECT_TRUE(document.find("missing") == nullptr);
}

void should_accept_nesting_up_to_the_limit() {
  const std::string nested_text = std::string(200, '[') + std::string(200, ']');
  EXPECT_TRUE(!throws_runtime_error(nested_text));
}

// --- negative ---

void should_reject_malformed_documents() {
  EXPECT_TRUE(throws_runtime_error(""));
  EXPECT_TRUE(throws_runtime_error("{\"a\": 1} trailing"));
  EXPECT_TRUE(throws_runtime_error("\"unterminated"));
  EXPECT_TRUE(throws_runtime_error("{\"a\" 1}"));
  EXPECT_TRUE(throws_runtime_error("[1, 2"));
  EXPECT_TRUE(throws_runtime_error("\"bad \\q escape\""));
}

void should_reject_nesting_beyond_the_limit() {
  EXPECT_TRUE(throws_runtime_error(std::string(1000, '[') + std::string(1000, ']')));
}

void should_throw_on_wrong_type_and_missing_member() {
  const JsonValue document = parse_json(R"({"count": 3})");
  bool has_thrown_for_type = false;
  bool has_thrown_for_member = false;
  try {
    document.at("count").as_string();
  } catch (const std::runtime_error&) {
    has_thrown_for_type = true;
  }
  try {
    document.at("absent");
  } catch (const std::runtime_error&) {
    has_thrown_for_member = true;
  }
  EXPECT_TRUE(has_thrown_for_type && has_thrown_for_member);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_parse_nested_objects_arrays_and_scalars),
      TEST_CASE(should_decode_string_escapes_including_surrogate_pairs),
      TEST_CASE(should_parse_empty_containers_and_surrounding_whitespace),
      TEST_CASE(should_accept_nesting_up_to_the_limit),
      TEST_CASE(should_reject_malformed_documents),
      TEST_CASE(should_reject_nesting_beyond_the_limit),
      TEST_CASE(should_throw_on_wrong_type_and_missing_member),
  });
}
