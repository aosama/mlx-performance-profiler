#pragma once

// Progress reporting: one concern only — periodic "done/total, elapsed, ETA"
// lines for any loop whose length is not known to be trivially short.

#include <chrono>
#include <string>

namespace mlx_plan {

// Linear extrapolation from the observed rate: crude, but honest about being
// an estimate, and it needs no model of per-unit cost. Returns 0 before any
// unit completes (no rate yet) and never goes negative, even if a caller
// over-reports completed units.
double estimate_remaining_seconds(
    double elapsed_seconds, int completed_units, int total_units);

class ProgressReporter {
 public:
  // Prints at most once per `report_interval_seconds` plus once at the end,
  // so a fast run stays quiet while a slow or hung one is never silent.
  ProgressReporter(
      std::string phase_label,
      int total_units,
      double report_interval_seconds = 5.0);

  void update(int completed_units);
  void finish(int completed_units);

 private:
  using Clock = std::chrono::steady_clock;
  void print_line(int completed_units) const;

  std::string phase_label_;
  int total_units_;
  double report_interval_seconds_;
  Clock::time_point started_at_;
  Clock::time_point last_printed_at_;
};

} // namespace mlx_plan
