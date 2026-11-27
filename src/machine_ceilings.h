#pragma once

// Machine ceilings: one concern only — measure, on the machine running the
// profile, how fast its GPU can read unified memory and how fast it can do
// matrix arithmetic, with MLX's own kernels. work_limits compares every
// operation with these two numbers.
//
// Why measured, not taken from a specification: the advertised memory
// bandwidth is a peak no kernel reaches, and the arithmetic a GPU delivers
// depends on the kernel MLX picks for it. A ceiling measured with MLX's own
// kernels is the one a job's kernels can actually be held to. It is a
// practical ceiling, not a physical one: a kernel may beat it (a quantized
// multiply can outrun the dense calibration multiply), and its share then
// reads above 100%.

#include <string>

namespace mlx_perf {

struct MachineCeilings {
  bool is_measured = false;
  double memory_read_bytes_per_second = 0.0;
  double arithmetic_operations_per_second = 0.0;
  // The type of the faster calibration multiply, e.g. "bfloat16".
  std::string arithmetic_dtype;
};

// Calibrates once per process, on first call, and returns the cached
// result: under a second and half a GiB of memory, released
// (and MLX's buffer pool cleared) before it returns. Not measured when the
// GPU gives no execution window.
const MachineCeilings& read_machine_ceilings();

} // namespace mlx_perf
