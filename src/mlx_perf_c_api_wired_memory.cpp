// The C entry points for wired memory (#45): one reading now, and the wire
// release probe. Shared C API plumbing lives in c_api_support.

#include "mlx_perf/mlx_perf_c_api.h"

#include "c_api_support.h"
#include "wire_release_probe.h"
#include "wired_memory_sampler.h"

#include <cmath>
#include <stdexcept>

namespace {

using namespace mlx_perf;

void read_wired_memory(mlx_perf_wired_memory* reading) {
  if (reading == nullptr) {
    throw std::invalid_argument("mlx_perf_read_wired_memory: null argument");
  }
  const WiredMemorySample wired_memory_sample = take_wired_memory_sample(WiredSamplePoint::interval);
  if (wired_memory_sample.system_wired_bytes == 0) {
    throw std::runtime_error("cannot read the system's wired memory");
  }
  reading->system_wired_bytes = wired_memory_sample.system_wired_bytes;
  reading->wire_limit_bytes = wired_memory_sample.wire_limit_bytes;
  reading->mlx_active_bytes = wired_memory_sample.mlx_active_bytes;
  reading->mlx_cache_bytes = wired_memory_sample.mlx_cache_bytes;
  reading->wired_outside_mlx_bytes = wired_outside_mlx_bytes(wired_memory_sample);
  reading->wire_headroom_bytes = wire_headroom_bytes(wired_memory_sample);
}

void measure_wire_release_lag(uint64_t allocation_bytes, int repetition_count, mlx_perf_wire_release_lag* result) {
  if (result == nullptr) {
    throw std::invalid_argument("mlx_perf_measure_wire_release_lag: null argument");
  }
  const WireReleaseProbeResult probe_result = mlx_perf::measure_wire_release_lag({allocation_bytes, repetition_count, 5.0});
  result->allocation_bytes = probe_result.allocation_bytes;
  result->wire_limit_bytes = probe_result.wire_limit_bytes;
  result->median_wired_rise_bytes = median_wired_rise_bytes(probe_result);
  result->median_release_lag_us = median_probe_release_lag_microseconds(probe_result).value_or(NAN);
  result->largest_release_lag_us = largest_probe_release_lag_microseconds(probe_result).value_or(NAN);
  result->repetition_count = static_cast<int>(probe_result.repetitions.size());
  result->resolved_count = count_resolved_repetitions(probe_result);
}

} // namespace

extern "C" int mlx_perf_read_wired_memory(mlx_perf_wired_memory* reading) {
  clear_last_c_api_error();
  try {
    read_wired_memory(reading);
    return 0;
  } catch (...) {
    return record_failure_in_flight();
  }
}

extern "C" int mlx_perf_measure_wire_release_lag(uint64_t allocation_bytes, int repetition_count,
                                                 mlx_perf_wire_release_lag* result) {
  clear_last_c_api_error();
  try {
    measure_wire_release_lag(allocation_bytes, repetition_count, result);
    return 0;
  } catch (...) {
    return record_failure_in_flight();
  }
}
