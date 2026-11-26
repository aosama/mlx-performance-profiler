#pragma once

// C API support: one concern only — what every C entry point shares: the
// thread's last error, turning the exception in flight into a C status,
// checking string arguments, and copying a judged metric into its C struct.
// The entry points themselves live in mlx_perf_c_api.cpp (graph jobs) and
// mlx_perf_c_api_procedures.cpp (procedure jobs).

#include "mlx_perf/mlx_perf_c_api.h"
#include "sample_statistics.h"

#include <initializer_list>

namespace mlx_perf {

void clear_last_c_api_error();

// The thread's last failure message, as mlx_perf_last_error() returns it.
const char* last_c_api_error();

// Called only from inside a catch block: records the exception in flight as
// the thread's last error and returns the C failure status (1).
int record_failure_in_flight();

// Every string an entry point reads must be present; the message names the
// entry point so a caller in another language can tell which call it got
// wrong.
void require_strings(std::initializer_list<const char*> required_strings, const char* entry_point_name);

void fill_metric_comparison(const PairedComparison& paired_comparison,
                            mlx_perf_metric_comparison& metric_comparison);

} // namespace mlx_perf
