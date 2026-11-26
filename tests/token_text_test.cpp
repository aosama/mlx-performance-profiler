// Unit tests for token_text: byte-level BPE decoding for the correctness check.

#include "qwen35/token_text.h"
#include "temporary_model_directory.h"
#include "test_harness.h"

#include <stdexcept>

namespace {

using namespace mlx_perf::qwen35;

// --- positive ---

void should_unmap_space_and_newline_markers() {
  EXPECT_TRUE(unmap_byte_level_text("\xC4\xA0Paris") == " Paris");  // Ġ → space
  EXPECT_TRUE(unmap_byte_level_text("\xC4\x8A") == "\n");           // Ċ → newline
  EXPECT_TRUE(unmap_byte_level_text("Hello") == "Hello");
}

void should_decode_ids_through_the_vocabulary() {
  const test_support::TemporaryModelDirectory model_directory;
  model_directory.write_file("vocab.json",
                             R"({"The": 760, "Ġcapital": 6511, "ĠParis": 11751})");
  const TokenText token_text(model_directory.path());
  EXPECT_TRUE(token_text.decode({760, 6511}) == "The capital");
  EXPECT_TRUE(token_text.find_token_id("\xC4\xA0Paris") == 11751);
}

// --- edge / boundary ---

void should_decode_an_empty_id_list_to_empty_text() {
  const test_support::TemporaryModelDirectory model_directory;
  model_directory.write_file("vocab.json", R"({"a": 0})");
  EXPECT_TRUE(TokenText(model_directory.path()).decode({}).empty());
}

void should_reassemble_a_multibyte_character_split_across_tokens() {
  // "é" is bytes C3 A9, byte-level-mapped to "Ã" and "©" — two tokens.
  const test_support::TemporaryModelDirectory model_directory;
  model_directory.write_file("vocab.json", R"({"Ã": 1, "©": 2})");
  EXPECT_TRUE(TokenText(model_directory.path()).decode({1, 2}) == "\xC3\xA9");
}

// --- negative ---

void should_render_unknown_ids_visibly_and_miss_unknown_entries() {
  const test_support::TemporaryModelDirectory model_directory;
  model_directory.write_file("vocab.json", R"({"a": 0})");
  const TokenText token_text(model_directory.path());
  EXPECT_TRUE(token_text.decode({0, 99, 0}) == "a<99>a");
  EXPECT_TRUE(token_text.find_token_id("missing") == -1);
}

void should_fail_clearly_without_a_vocabulary_file() {
  bool has_thrown = false;
  try {
    TokenText("/nonexistent/model/directory");
  } catch (const std::runtime_error&) {
    has_thrown = true;
  }
  EXPECT_TRUE(has_thrown);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_unmap_space_and_newline_markers),
      TEST_CASE(should_decode_ids_through_the_vocabulary),
      TEST_CASE(should_decode_an_empty_id_list_to_empty_text),
      TEST_CASE(should_reassemble_a_multibyte_character_split_across_tokens),
      TEST_CASE(should_render_unknown_ids_visibly_and_miss_unknown_entries),
      TEST_CASE(should_fail_clearly_without_a_vocabulary_file),
  });
}
