#pragma once

// Token text: one concern only — turn token ids back into text for a
// human-readable correctness check. Decoding only (no tokenizer): Qwen's
// byte-level BPE vocabulary maps each byte to a printable code point
// (GPT-2's bytes_to_unicode), so decoding is "concatenate, then un-map".

#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_plan::qwen35 {

class TokenText {
 public:
  // Reads <model_directory>/vocab.json. Throws std::runtime_error on failure.
  explicit TokenText(const std::string& model_directory);
  // Tokens absent from the vocabulary (special tokens) render as "<id>".
  std::string decode(const std::vector<int>& token_ids) const;
  // The id of an exact vocabulary entry (e.g. "ĠParis"), or -1.
  int find_token_id(const std::string& vocabulary_entry) const;

 private:
  std::unordered_map<int, std::string> token_id_to_entry_;
  std::unordered_map<std::string, int> entry_to_token_id_;
};

// GPT-2 bytes_to_unicode, inverted: printable code point → original byte.
// Exposed for tests.
std::string unmap_byte_level_text(const std::string& vocabulary_text);

} // namespace mlx_plan::qwen35
