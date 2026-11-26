#include "console_output.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>

namespace mlx_perf {
namespace {

// Atomic: a caller may switch output from one thread while another measures.
std::atomic<bool> console_output_enabled{true};

void print_to(std::FILE* output_stream, const char* format, std::va_list format_arguments) {
  std::vfprintf(output_stream, format, format_arguments);
  std::fflush(output_stream);
}

} // namespace

void set_console_output_enabled(bool is_enabled) { console_output_enabled.store(is_enabled); }

bool is_console_output_enabled() { return console_output_enabled.load(); }

void print_progress_line(const char* format, ...) {
  if (!is_console_output_enabled()) {
    return;
  }
  std::va_list format_arguments;
  va_start(format_arguments, format);
  print_to(stdout, format, format_arguments);
  va_end(format_arguments);
}

void print_error_line(const char* format, ...) {
  if (!is_console_output_enabled()) {
    return;
  }
  std::va_list format_arguments;
  va_start(format_arguments, format);
  print_to(stderr, format, format_arguments);
  va_end(format_arguments);
}

} // namespace mlx_perf
