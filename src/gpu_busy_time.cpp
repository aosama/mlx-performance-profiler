#include "gpu_busy_time.h"

#include <algorithm>

namespace mlx_perf {

double measure_gpu_busy_microseconds(std::vector<GpuWindow> gpu_windows,
                                     double span_started_seconds, double span_finished_seconds) {
  if (span_finished_seconds <= span_started_seconds) {
    return 0.0;
  }
  std::sort(gpu_windows.begin(), gpu_windows.end(),
            [](const GpuWindow& earlier, const GpuWindow& later) {
              return earlier.started_seconds < later.started_seconds;
            });
  double busy_seconds = 0.0;
  double covered_until_seconds = span_started_seconds;
  for (const GpuWindow& gpu_window : gpu_windows) {
    const double started = std::max(gpu_window.started_seconds, covered_until_seconds);
    const double finished = std::min(gpu_window.finished_seconds, span_finished_seconds);
    if (finished > started) {
      busy_seconds += finished - started;
      covered_until_seconds = finished;
    }
  }
  return busy_seconds * 1e6;
}

} // namespace mlx_perf
