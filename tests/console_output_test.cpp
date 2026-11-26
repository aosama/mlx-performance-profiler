// Unit tests for console_output: the switch, and that switched-off output
// reaches neither stdout nor stderr.

#include "captured_console.h"
#include "console_output.h"
#include "test_harness.h"

#include <string>

namespace {

using captured_console::CapturedConsole;
using mlx_perf::print_error_line;
using mlx_perf::print_progress_line;
using mlx_perf::set_console_output_enabled;

// --- positive ---

void should_print_progress_to_stdout_and_errors_to_stderr_by_default() {
  EXPECT_TRUE(mlx_perf::is_console_output_enabled());
  CapturedConsole captured_console("console_output_default");
  print_progress_line("progress %d of %d\n", 3, 7);
  print_error_line("error: %s\n", "no such directory");
  EXPECT_TRUE(captured_console.stdout_text() == "progress 3 of 7\n");
  EXPECT_TRUE(captured_console.stderr_text() == "error: no such directory\n");
}

// --- edge / boundary ---

void should_resume_printing_once_switched_back_on() {
  set_console_output_enabled(false);
  set_console_output_enabled(true);
  CapturedConsole captured_console("console_output_resumed");
  print_progress_line("back\n");
  EXPECT_TRUE(captured_console.stdout_text() == "back\n");
}

// --- negative: switched off, nothing escapes ---

void should_print_nothing_anywhere_when_switched_off() {
  set_console_output_enabled(false);
  CapturedConsole captured_console("console_output_silenced");
  print_progress_line("progress %d\n", 1);
  print_error_line("error: %s\n", "hidden");
  const std::string stdout_text = captured_console.stdout_text();
  const std::string stderr_text = captured_console.stderr_text();
  set_console_output_enabled(true);
  EXPECT_TRUE(stdout_text.empty());
  EXPECT_TRUE(stderr_text.empty());
  EXPECT_TRUE(mlx_perf::is_console_output_enabled());
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_print_progress_to_stdout_and_errors_to_stderr_by_default),
      TEST_CASE(should_resume_printing_once_switched_back_on),
      TEST_CASE(should_print_nothing_anywhere_when_switched_off),
  });
}
