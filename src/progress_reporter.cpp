#include "progress_reporter.h"

#include <cstdio>
#include <utility>

namespace mlx_plan {

double estimate_remaining_seconds(
    double elapsed_seconds, int completed_units, int total_units) {
  if (completed_units <= 0 || completed_units >= total_units) {
    return 0.0;
  }
  return elapsed_seconds * (total_units - completed_units) / completed_units;
}

ProgressReporter::ProgressReporter(
    std::string phase_label,
    int total_units,
    double report_interval_seconds)
    : phase_label_(std::move(phase_label)),
      total_units_(total_units),
      report_interval_seconds_(report_interval_seconds),
      started_at_(Clock::now()),
      last_printed_at_(started_at_) {}

void ProgressReporter::update(int completed_units) {
  const double seconds_since_last_print =
      std::chrono::duration<double>(Clock::now() - last_printed_at_).count();
  if (seconds_since_last_print >= report_interval_seconds_) {
    print_line(completed_units);
    last_printed_at_ = Clock::now();
  }
}

void ProgressReporter::finish(int completed_units) {
  print_line(completed_units);
}

void ProgressReporter::print_line(int completed_units) const {
  const double elapsed_seconds =
      std::chrono::duration<double>(Clock::now() - started_at_).count();
  const double eta_seconds =
      estimate_remaining_seconds(elapsed_seconds, completed_units, total_units_);
  const double percent_complete =
      total_units_ > 0 ? 100.0 * completed_units / total_units_ : 100.0;
  std::printf(
      "  %s: %d/%d (%.0f%%)  elapsed %.2fs  eta %.2fs\n",
      phase_label_.c_str(),
      completed_units,
      total_units_,
      percent_complete,
      elapsed_seconds,
      eta_seconds);
  // stdout is fully buffered when piped (tee, CI, a Rust parent process);
  // without a flush, "live" progress would only appear at exit.
  std::fflush(stdout);
}

} // namespace mlx_plan
