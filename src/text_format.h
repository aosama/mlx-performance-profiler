#pragma once

// Text formatting: one concern only — human-readable durations and sizes,
// and escaping for the two text formats the report emits (HTML, DOT).
// Shared so the console, the DAG and the HTML report never disagree on units.

#include <cstddef>
#include <string>

namespace mlx_plan {

// "412.3 µs" below a millisecond, "1.24 ms" at or above.
std::string format_microseconds(double duration_microseconds);
// "512 B", "3.5 KiB", "1.2 MiB".
std::string format_bytes(std::size_t byte_count);
std::string escape_html_text(const std::string& raw_text);
// Escapes quotes and backslashes; turns real newlines into DOT line breaks.
std::string escape_dot_text(const std::string& raw_text);

} // namespace mlx_plan
