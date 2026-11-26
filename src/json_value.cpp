#include "json_value.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mlx_perf {
namespace {

// Nesting limit: a hostile or corrupt file must not overflow the stack.
constexpr int MAX_NESTING_DEPTH = 256;

class JsonParser {
 public:
  explicit JsonParser(const std::string& json_text) : json_text_(json_text) {}

  JsonValue parse_document() {
    JsonValue document = parse_value(0);
    skip_whitespace();
    if (position_ != json_text_.size()) {
      fail("trailing characters after the document");
    }
    return document;
  }

 private:
  [[noreturn]] void fail(const std::string& reason) const {
    throw std::runtime_error("JSON parse error at byte " + std::to_string(position_) +
                             ": " + reason);
  }

  void skip_whitespace() {
    while (position_ < json_text_.size() &&
           (json_text_[position_] == ' ' || json_text_[position_] == '\n' ||
            json_text_[position_] == '\r' || json_text_[position_] == '\t')) {
      ++position_;
    }
  }

  char peek() const { return position_ < json_text_.size() ? json_text_[position_] : '\0'; }

  void expect(char expected_letter) {
    if (peek() != expected_letter) {
      fail(std::string("expected '") + expected_letter + "'");
    }
    ++position_;
  }

  bool consume_literal(const char* literal_text) {
    const std::string literal(literal_text);
    if (json_text_.compare(position_, literal.size(), literal) != 0) {
      return false;
    }
    position_ += literal.size();
    return true;
  }

  JsonValue parse_value(int nesting_depth) {
    if (nesting_depth > MAX_NESTING_DEPTH) {
      fail("nesting too deep");
    }
    skip_whitespace();
    const char leading_letter = peek();
    if (leading_letter == '{') return parse_object(nesting_depth);
    if (leading_letter == '[') return parse_array(nesting_depth);
    if (leading_letter == '"') return JsonValue(parse_string());
    if (consume_literal("true")) return JsonValue(true);
    if (consume_literal("false")) return JsonValue(false);
    if (consume_literal("null")) return JsonValue();
    return JsonValue(parse_number());
  }

  JsonValue parse_object(int nesting_depth) {
    expect('{');
    JsonValue::Object members;
    skip_whitespace();
    if (peek() == '}') {
      ++position_;
      return JsonValue(std::move(members));
    }
    while (true) {
      skip_whitespace();
      std::string member_name = parse_string();
      skip_whitespace();
      expect(':');
      members[member_name] = parse_value(nesting_depth + 1);
      skip_whitespace();
      if (peek() == '}') {
        ++position_;
        return JsonValue(std::move(members));
      }
      expect(',');
    }
  }

  JsonValue parse_array(int nesting_depth) {
    expect('[');
    JsonValue::Array elements;
    skip_whitespace();
    if (peek() == ']') {
      ++position_;
      return JsonValue(std::move(elements));
    }
    while (true) {
      elements.push_back(parse_value(nesting_depth + 1));
      skip_whitespace();
      if (peek() == ']') {
        ++position_;
        return JsonValue(std::move(elements));
      }
      expect(',');
    }
  }

  double parse_number() {
    const char* number_start = json_text_.c_str() + position_;
    char* number_end = nullptr;
    const double number_value = std::strtod(number_start, &number_end);
    if (number_end == number_start) {
      fail("expected a value");
    }
    position_ += static_cast<std::size_t>(number_end - number_start);
    return number_value;
  }

  void append_utf8(std::string& text, unsigned code_point) {
    if (code_point < 0x80) {
      text += static_cast<char>(code_point);
    } else if (code_point < 0x800) {
      text += static_cast<char>(0xC0 | (code_point >> 6));
      text += static_cast<char>(0x80 | (code_point & 0x3F));
    } else if (code_point < 0x10000) {
      text += static_cast<char>(0xE0 | (code_point >> 12));
      text += static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
      text += static_cast<char>(0x80 | (code_point & 0x3F));
    } else {
      text += static_cast<char>(0xF0 | (code_point >> 18));
      text += static_cast<char>(0x80 | ((code_point >> 12) & 0x3F));
      text += static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
      text += static_cast<char>(0x80 | (code_point & 0x3F));
    }
  }

  unsigned parse_hex_quad() {
    if (position_ + 4 > json_text_.size()) {
      fail("truncated \\u escape");
    }
    const std::string hex_digits = json_text_.substr(position_, 4);
    char* digits_end = nullptr;
    const unsigned long code_unit = std::strtoul(hex_digits.c_str(), &digits_end, 16);
    if (digits_end != hex_digits.c_str() + 4) {
      fail("bad \\u escape");
    }
    position_ += 4;
    return static_cast<unsigned>(code_unit);
  }

  // 😀-style surrogate pairs combine into one code point.
  unsigned parse_unicode_escape() {
    unsigned code_point = parse_hex_quad();
    if (code_point >= 0xD800 && code_point <= 0xDBFF && consume_literal("\\u")) {
      const unsigned low_surrogate = parse_hex_quad();
      code_point = 0x10000 + ((code_point - 0xD800) << 10) + (low_surrogate - 0xDC00);
    }
    return code_point;
  }

  void append_escape(std::string& text) {
    const char escape_letter = peek();
    ++position_;
    switch (escape_letter) {
      case '"': text += '"'; break;
      case '\\': text += '\\'; break;
      case '/': text += '/'; break;
      case 'b': text += '\b'; break;
      case 'f': text += '\f'; break;
      case 'n': text += '\n'; break;
      case 'r': text += '\r'; break;
      case 't': text += '\t'; break;
      case 'u': append_utf8(text, parse_unicode_escape()); break;
      default: fail("unknown escape");
    }
  }

  std::string parse_string() {
    expect('"');
    std::string text;
    while (true) {
      if (position_ >= json_text_.size()) {
        fail("unterminated string");
      }
      const char letter = json_text_[position_++];
      if (letter == '"') {
        return text;
      }
      if (letter == '\\') {
        append_escape(text);
      } else {
        text += letter;
      }
    }
  }

  const std::string& json_text_;
  std::size_t position_ = 0;
};

[[noreturn]] void throw_type_error(const char* expected_type) {
  throw std::runtime_error(std::string("JSON value is not ") + expected_type);
}

} // namespace

bool JsonValue::is_object() const {
  return std::holds_alternative<std::shared_ptr<Object>>(value_);
}

bool JsonValue::is_array() const {
  return std::holds_alternative<std::shared_ptr<Array>>(value_);
}

double JsonValue::as_number() const {
  if (!is_number()) throw_type_error("a number");
  return std::get<double>(value_);
}

bool JsonValue::as_bool() const {
  if (!std::holds_alternative<bool>(value_)) throw_type_error("a boolean");
  return std::get<bool>(value_);
}

const std::string& JsonValue::as_string() const {
  if (!is_string()) throw_type_error("a string");
  return std::get<std::string>(value_);
}

const JsonValue::Array& JsonValue::as_array() const {
  if (!is_array()) throw_type_error("an array");
  return *std::get<std::shared_ptr<Array>>(value_);
}

const JsonValue::Object& JsonValue::as_object() const {
  if (!is_object()) throw_type_error("an object");
  return *std::get<std::shared_ptr<Object>>(value_);
}

const JsonValue* JsonValue::find(const std::string& member_name) const {
  const Object& members = as_object();
  const auto member = members.find(member_name);
  return member == members.end() ? nullptr : &member->second;
}

const JsonValue& JsonValue::at(const std::string& member_name) const {
  const JsonValue* member = find(member_name);
  if (member == nullptr) {
    throw std::runtime_error("JSON object has no member \"" + member_name + "\"");
  }
  return *member;
}

JsonValue parse_json(const std::string& json_text) {
  return JsonParser(json_text).parse_document();
}

JsonValue parse_json_file(const std::string& file_path) {
  std::ifstream json_file(file_path, std::ios::binary);
  if (!json_file) {
    throw std::runtime_error("cannot open " + file_path);
  }
  std::ostringstream file_text;
  file_text << json_file.rdbuf();
  return parse_json(file_text.str());
}

} // namespace mlx_perf
