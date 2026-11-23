#include "text_format.h"

#include <cstdio>

namespace mlx_plan {
namespace {

constexpr double MICROSECONDS_PER_MILLISECOND = 1000.0;
constexpr double BYTES_PER_KIBIBYTE = 1024.0;
constexpr double BYTES_PER_MEBIBYTE = 1024.0 * 1024.0;

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
  if (byte_amount >= BYTES_PER_MEBIBYTE) {
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

} // namespace mlx_plan
