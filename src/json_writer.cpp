#include "json_writer.h"

#include <charconv>
#include <cmath>
#include <cstdio>

namespace mlx_perf {
namespace {

constexpr int INDENT_SPACES_PER_LEVEL = 2;

void append_indent(std::string& json_text, int nesting_depth) {
  json_text.append(static_cast<std::size_t>(nesting_depth * INDENT_SPACES_PER_LEVEL), ' ');
}

// std::to_chars without a precision gives the shortest text that parses back
// to the same double ("12.5", "3", "1e-06"), unlike printf's %.17g, which
// writes 12.499999999999998 for values a reader expects to be round.
void append_number(std::string& json_text, double number_value) {
  if (!std::isfinite(number_value)) {
    json_text += "null";
    return;
  }
  char number_text[32];
  const auto conversion =
      std::to_chars(number_text, number_text + sizeof(number_text), number_value);
  json_text.append(number_text, conversion.ptr);
}

void append_string(std::string& json_text, const std::string& text_value) {
  json_text += '"';
  for (const char letter : text_value) {
    switch (letter) {
      case '"': json_text += "\\\""; break;
      case '\\': json_text += "\\\\"; break;
      case '\n': json_text += "\\n"; break;
      case '\r': json_text += "\\r"; break;
      case '\t': json_text += "\\t"; break;
      default:
        if (static_cast<unsigned char>(letter) < 0x20) {
          char escaped_control[8];
          std::snprintf(escaped_control, sizeof(escaped_control), "\\u%04x",
                        static_cast<unsigned>(static_cast<unsigned char>(letter)));
          json_text += escaped_control;
        } else {
          json_text += letter;  // UTF-8 bytes pass through unchanged
        }
    }
  }
  json_text += '"';
}

void append_value(std::string& json_text, const JsonValue& json_value, int nesting_depth);

void append_array(std::string& json_text, const JsonValue::Array& array_values,
                  int nesting_depth) {
  if (array_values.empty()) {
    json_text += "[]";
    return;
  }
  json_text += "[\n";
  for (std::size_t element_index = 0; element_index < array_values.size(); ++element_index) {
    append_indent(json_text, nesting_depth + 1);
    append_value(json_text, array_values[element_index], nesting_depth + 1);
    json_text += element_index + 1 < array_values.size() ? ",\n" : "\n";
  }
  append_indent(json_text, nesting_depth);
  json_text += ']';
}

void append_object(std::string& json_text, const JsonValue::Object& object_members,
                   int nesting_depth) {
  if (object_members.empty()) {
    json_text += "{}";
    return;
  }
  json_text += "{\n";
  std::size_t written_member_count = 0;
  for (const auto& [member_name, member_value] : object_members) {
    append_indent(json_text, nesting_depth + 1);
    append_string(json_text, member_name);
    json_text += ": ";
    append_value(json_text, member_value, nesting_depth + 1);
    ++written_member_count;
    json_text += written_member_count < object_members.size() ? ",\n" : "\n";
  }
  append_indent(json_text, nesting_depth);
  json_text += '}';
}

void append_value(std::string& json_text, const JsonValue& json_value, int nesting_depth) {
  if (json_value.is_null()) {
    json_text += "null";
  } else if (json_value.is_bool()) {
    json_text += json_value.as_bool() ? "true" : "false";
  } else if (json_value.is_number()) {
    append_number(json_text, json_value.as_number());
  } else if (json_value.is_string()) {
    append_string(json_text, json_value.as_string());
  } else if (json_value.is_array()) {
    append_array(json_text, json_value.as_array(), nesting_depth);
  } else {
    append_object(json_text, json_value.as_object(), nesting_depth);
  }
}

} // namespace

std::string write_json(const JsonValue& document) {
  std::string json_text;
  append_value(json_text, document, 0);
  json_text += '\n';
  return json_text;
}

} // namespace mlx_perf
