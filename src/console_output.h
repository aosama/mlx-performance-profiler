#pragma once

// Console output: one concern only — the progress, file and error lines the
// library prints, and the process-wide switch that silences them. Every line
// the library writes goes through here, so a caller that turns output off
// gets none, whichever module would have printed.

namespace mlx_perf {

// On by default: the tools print their progress. A library consumer whose
// own output must stay machine-readable turns it off. Thread-safe.
void set_console_output_enabled(bool is_enabled);
bool is_console_output_enabled();

// printf-style lines. Progress goes to stdout, flushed at once so it stays
// live when stdout is a pipe; errors go to stderr. Both write nothing while
// output is off; failures still reach callers through return values.
void print_progress_line(const char* format, ...) __attribute__((format(printf, 1, 2)));
void print_error_line(const char* format, ...) __attribute__((format(printf, 1, 2)));

} // namespace mlx_perf
