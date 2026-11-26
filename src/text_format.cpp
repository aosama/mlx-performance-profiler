#include "text_format.h"

#include <cctype>

#include <cstdio>

namespace mlx_perf {
namespace {

constexpr double MICROSECONDS_PER_MILLISECOND = 1000.0;
constexpr double BYTES_PER_KIBIBYTE = 1024.0;
constexpr double BYTES_PER_MEBIBYTE = 1024.0 * 1024.0;
constexpr double BYTES_PER_GIBIBYTE = 1024.0 * BYTES_PER_MEBIBYTE;

} // namespace

std::string format_microseconds(double duration_microseconds) {
  char formatted_duration[48];
  if (duration_microseconds >= MICROSECONDS_PER_MILLISECOND) {
    std::snprintf(
        formatted_duration, sizeof(formatted_duration), "%.2f ms",
        duration_microseconds / MICROSECONDS_PER_MILLISECOND);
  } else {
    std::snprintf(
        formatted_duration, sizeof(formatted_duration), "%.1f µs",
        duration_microseconds);
  }
  return formatted_duration;
}

std::string format_bytes(std::size_t byte_count) {
  char formatted_size[48];
  const double byte_amount = static_cast<double>(byte_count);
  if (byte_amount >= BYTES_PER_GIBIBYTE) {
    // Two decimals: wired memory and its limit are tens of GiB, and a
    // release of a few hundred MiB must still show.
    std::snprintf(
        formatted_size, sizeof(formatted_size), "%.2f GiB",
        byte_amount / BYTES_PER_GIBIBYTE);
  } else if (byte_amount >= BYTES_PER_MEBIBYTE) {
    std::snprintf(
        formatted_size, sizeof(formatted_size), "%.1f MiB",
        byte_amount / BYTES_PER_MEBIBYTE);
  } else if (byte_amount >= BYTES_PER_KIBIBYTE) {
    std::snprintf(
        formatted_size, sizeof(formatted_size), "%.1f KiB",
        byte_amount / BYTES_PER_KIBIBYTE);
  } else {
    std::snprintf(formatted_size, sizeof(formatted_size), "%zu B", byte_count);
  }
  return formatted_size;
}

std::string escape_html_text(const std::string& raw_text) {
  std::string escaped_text;
  escaped_text.reserve(raw_text.size());
  for (const char letter : raw_text) {
    switch (letter) {
      case '&': escaped_text += "&amp;"; break;
      case '<': escaped_text += "&lt;"; break;
      case '>': escaped_text += "&gt;"; break;
      case '"': escaped_text += "&quot;"; break;
      case '\'': escaped_text += "&#39;"; break;
      default: escaped_text += letter;
    }
  }
  return escaped_text;
}

std::string escape_json_text(const std::string& raw_text) {
  std::string escaped_text;
  escaped_text.reserve(raw_text.size());
  for (const char letter : raw_text) {
    switch (letter) {
      case '"': escaped_text += "\\\""; break;
      case '\\': escaped_text += "\\\\"; break;
      case '\n': escaped_text += "\\n"; break;
      case '\r': escaped_text += "\\r"; break;
      case '\t': escaped_text += "\\t"; break;
      default:
        if (static_cast<unsigned char>(letter) < 0x20) {
          char unicode_escape[8];
          std::snprintf(unicode_escape, sizeof(unicode_escape), "\\u%04x",
                        static_cast<unsigned char>(letter));
          escaped_text += unicode_escape;
        } else {
          escaped_text += letter;
        }
    }
  }
  return escaped_text;
}

std::string escape_dot_text(const std::string& raw_text) {
  std::string escaped_text;
  escaped_text.reserve(raw_text.size());
  for (const char letter : raw_text) {
    // A real newline becomes DOT's own line-break escape, so callers build
    // multi-line labels with '\n' and never hand-write DOT escapes.
    if (letter == '\n') {
      escaped_text += "\\n";
      continue;
    }
    if (letter == '"' || letter == '\\') {
      escaped_text += '\\';
    }
    escaped_text += letter;
  }
  return escaped_text;
}

std::string encode_url_path_segment(const std::string& raw_text) {
  static constexpr char HEX_DIGITS[] = "0123456789ABCDEF";
  std::string encoded_text;
  for (const char letter : raw_text) {
    const unsigned char byte_value = static_cast<unsigned char>(letter);
    const bool is_unreserved = std::isalnum(byte_value) || letter == '-' || letter == '.' ||
                               letter == '_' || letter == '~';
    if (is_unreserved) {
      encoded_text += letter;
    } else {
      encoded_text += '%';
      encoded_text += HEX_DIGITS[byte_value >> 4];
      encoded_text += HEX_DIGITS[byte_value & 0x0F];
    }
  }
  return encoded_text;
}

bool are_equal_ignoring_case(const std::string& first_name, const std::string& second_name) {
  if (first_name.size() != second_name.size()) {
    return false;
  }
  for (std::size_t letter_index = 0; letter_index < first_name.size(); ++letter_index) {
    if (std::tolower(static_cast<unsigned char>(first_name[letter_index])) !=
        std::tolower(static_cast<unsigned char>(second_name[letter_index]))) {
      return false;
    }
  }
  return true;
}

} // namespace mlx_perf
