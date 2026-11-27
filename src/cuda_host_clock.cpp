#include "gpu_host_clock.h"

#include "cupti_session.h"

namespace mlx_perf {

// The session's clock, read directly: CUPTI stamps every record it delivers
// with it (cupti_session), so a host instant read here and a GPU start read
// from a record compare with no conversion. Reading it does not start CUPTI.
double current_gpu_host_seconds() {
  return static_cast<double>(current_session_nanoseconds()) / 1e9;
}

} // namespace mlx_perf
