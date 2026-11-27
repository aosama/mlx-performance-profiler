#pragma once

// Text formatting: one concern only — human-readable durations and sizes,
// and escaping for the two text formats the report emits (HTML, DOT).
// Shared so the console, the DAG and the HTML report never disagree on units.

#include <cstddef>
#include <string>

namespace mlx_perf {

// "412.3 µs" below a millisecond, "1.24 ms" at or above.
std::string format_microseconds(double duration_microseconds);
// "512 B", "3.5 KiB", "1.2 MiB".
std::string format_bytes(std::size_t byte_count);
// Arithmetic, in operations (two per multiply-add): "820 M", "1.37 G", "4.40 T".
std::string format_operation_count(double operation_count);
// "412.5 GiB/s": a reading speed, in the same binary units as format_bytes.
std::string format_bytes_per_second(double bytes_per_second);
// "23.4 TFLOP/s" at or above a trillion, "512.0 GFLOP/s" below.
std::string format_operations_per_second(double operations_per_second);
std::string escape_html_text(const std::string& raw_text);
// For a JSON string body: quotes, backslashes and control characters.
std::string escape_json_text(const std::string& raw_text);
// Percent-encodes everything but RFC 3986's unreserved characters, so a
// caller-chosen file name ("v2#fast", "50%") works as a relative link.
std::string encode_url_path_segment(const std::string& raw_text);

// True when two names would name the same file on a case-insensitive file
// system (macOS's default): ASCII letters compare without case.
bool are_equal_ignoring_case(const std::string& first_name, const std::string& second_name);

// Escapes quotes and backslashes; turns real newlines into DOT line breaks.
std::string escape_dot_text(const std::string& raw_text);

} // namespace mlx_perf
