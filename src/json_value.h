#pragma once

// JSON value: one concern only — parse a JSON document into a tree. MLX's
// installed headers carry no JSON library, and model directories describe
// themselves in config.json / vocab.json; a ~200-line recursive-descent
// parser is cheaper than a fetched dependency for read-only configs.

#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace mlx_perf {

class JsonValue {
 public:
  using Array = std::vector<JsonValue>;
  using Object = std::map<std::string, JsonValue>;

  JsonValue() = default;  // null
  explicit JsonValue(bool flag_value) : value_(flag_value) {}
  explicit JsonValue(double number_value) : value_(number_value) {}
  explicit JsonValue(std::string text_value) : value_(std::move(text_value)) {}
  explicit JsonValue(Array array_value)
      : value_(std::make_shared<Array>(std::move(array_value))) {}
  explicit JsonValue(Object object_value)
      : value_(std::make_shared<Object>(std::move(object_value))) {}

  bool is_null() const { return std::holds_alternative<std::monostate>(value_); }
  bool is_bool() const { return std::holds_alternative<bool>(value_); }
  bool is_number() const { return std::holds_alternative<double>(value_); }
  bool is_string() const { return std::holds_alternative<std::string>(value_); }
  bool is_object() const;
  bool is_array() const;

  // Typed reads throw std::runtime_error naming the expected type, so a
  // malformed config fails with a reason instead of a wrong number.
  double as_number() const;
  bool as_bool() const;
  const std::string& as_string() const;
  const Array& as_array() const;
  const Object& as_object() const;

  // Object member access; throws when absent. `find` returns nullptr instead.
  const JsonValue& at(const std::string& member_name) const;
  const JsonValue* find(const std::string& member_name) const;

 private:
  std::variant<std::monostate, bool, double, std::string, std::shared_ptr<Array>,
               std::shared_ptr<Object>>
      value_;
};

// Throws std::runtime_error with the byte offset on malformed input.
JsonValue parse_json(const std::string& json_text);
JsonValue parse_json_file(const std::string& file_path);

} // namespace mlx_perf
