#pragma once

// GPU device facts: one concern only — the facts about the GPU that turn a
// kernel launch's shape into an occupancy figure: how many cores it has, and
// its name for the report.

#include <string>

namespace mlx_perf {

struct GpuDeviceFacts {
  std::string gpu_name;   // e.g. "Apple M5 Pro"; empty when unknown
  int gpu_core_count = 0; // 0 when the platform does not report it
};

// Read once per process and cached: the GPU cannot change while it runs.
// The core count comes from the IO registry's "gpu-core-count" on the Apple
// GPU's accelerator service, the figure Apple's own tools show; Metal's API
// does not expose it.
const GpuDeviceFacts& read_gpu_device_facts();

} // namespace mlx_perf
