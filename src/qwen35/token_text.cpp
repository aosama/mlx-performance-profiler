#include "token_text.h"

#include "json_value.h"

#include <array>

namespace mlx_plan::qwen35 {
namespace {

// bytes_to_unicode: printable Latin-1 bytes map to themselves; the other 68
// bytes map to code points 256, 257, ... in byte order.
std::unordered_map<unsigned, unsigned char> build_code_point_to_byte() {
  std::array<bool, 256> is_printable{};
  for (unsigned byte_value = 0; byte_value < 256; ++byte_value) {
    is_printable[byte_value] = (byte_value >= '!' && byte_value <= '~') ||
        (byte_value >= 0xA1 && byte_value <= 0xAC) || (byte_value >= 0xAE);
  }
  std::unordered_map<unsigned, unsigned char> code_point_to_byte;
  unsigned shifted_code_point = 256;
  for (unsigned byte_value = 0; byte_value < 256; ++byte_value) {
    const unsigned code_point = is_printable[byte_value] ? byte_value : shifted_code_point++;
    code_point_to_byte[code_point] = static_cast<unsigned char>(byte_value);
  }
  return code_point_to_byte;
}

// Reads one UTF-8 code point starting at `text_position`, advancing it.
unsigned read_code_point(const std::string& text, std::size_t& text_position) {
  const unsigned char lead_byte = static_cast<unsigned char>(text[text_position++]);
  int continuation_count = lead_byte >= 0xF0 ? 3 : lead_byte >= 0xE0 ? 2 : lead_byte >= 0xC0 ? 1 : 0;
  unsigned code_point = continuation_count == 0 ? lead_byte
                        : lead_byte & (0x3F >> continuation_count);
  while (continuation_count-- > 0 && text_position < text.size()) {
    code_point = (code_point << 6) | (static_cast<unsigned char>(text[text_position++]) & 0x3F);
  }
  return code_point;
}

} // namespace

std::string unmap_byte_level_text(const std::string& vocabulary_text) {
  static const auto code_point_to_byte = build_code_point_to_byte();
  std::string original_bytes;
  std::size_t text_position = 0;
  while (text_position < vocabulary_text.size()) {
    const unsigned code_point = read_code_point(vocabulary_text, text_position);
    const auto byte_entry = code_point_to_byte.find(code_point);
    // A code point outside the map is not byte-level text; keep it as '?'.
    original_bytes += byte_entry == code_point_to_byte.end()
        ? '?'
        : static_cast<char>(byte_entry->second);
  }
  return original_bytes;
}

TokenText::TokenText(const std::string& model_directory) {
  // Bound to a local: a range-for over a member of a temporary dangles.
  const JsonValue vocabulary = parse_json_file(model_directory + "/vocab.json");
  for (const auto& [vocabulary_entry, token_id] : vocabulary.as_object()) {
    const int id = static_cast<int>(token_id.as_number());
    token_id_to_entry_[id] = vocabulary_entry;
    entry_to_token_id_[vocabulary_entry] = id;
  }
}

std::string TokenText::decode(const std::vector<int>& token_ids) const {
  std::string vocabulary_text;
  std::string decoded_text;
  for (const int token_id : token_ids) {
    const auto entry = token_id_to_entry_.find(token_id);
    if (entry == token_id_to_entry_.end()) {
      decoded_text += unmap_byte_level_text(vocabulary_text) + "<" + std::to_string(token_id) + ">";
      vocabulary_text.clear();
    } else {
      vocabulary_text += entry->second;
    }
  }
  return decoded_text + unmap_byte_level_text(vocabulary_text);
}

int TokenText::find_token_id(const std::string& vocabulary_entry) const {
  const auto entry = entry_to_token_id_.find(vocabulary_entry);
  return entry == entry_to_token_id_.end() ? -1 : entry->second;
}

} // namespace mlx_plan::qwen35
