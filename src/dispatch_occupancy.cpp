#include "dispatch_occupancy.h"

#include <algorithm>

namespace mlx_perf {
namespace {

std::uint64_t divide_rounding_up(std::uint64_t thread_count, std::uint64_t threads_per_group) {
  return threads_per_group == 0 ? 0 : (thread_count + threads_per_group - 1) / threads_per_group;
}

} // namespace

std::uint64_t grid_volume(const GridSize& grid) {
  return grid.width * grid.height * grid.depth;
}

GridSize threadgroup_grid(const KernelDispatch& kernel_dispatch) {
  if (kernel_dispatch.grid_unit == DispatchGridUnit::threadgroups) {
    return kernel_dispatch.grid;
  }
  const GridSize& threads = kernel_dispatch.grid;
  const GridSize& group = kernel_dispatch.threads_per_threadgroup;
  return GridSize{divide_rounding_up(threads.width, group.width),
                  divide_rounding_up(threads.height, group.height),
                  divide_rounding_up(threads.depth, group.depth)};
}

std::uint64_t count_threadgroups(const KernelDispatch& kernel_dispatch) {
  if (grid_volume(kernel_dispatch.threads_per_threadgroup) == 0) {
    return 0;
  }
  return grid_volume(threadgroup_grid(kernel_dispatch));
}

double gpu_core_coverage(std::uint64_t threadgroup_count, int gpu_core_count) {
  if (gpu_core_count <= 0) {
    return -1.0;
  }
  return std::min(1.0, static_cast<double>(threadgroup_count) / gpu_core_count);
}

bool can_fill_gpu(std::uint64_t threadgroup_count, int gpu_core_count) {
  return gpu_core_count > 0 && threadgroup_count >= static_cast<std::uint64_t>(gpu_core_count);
}

const char* dispatch_grid_unit_name(DispatchGridUnit grid_unit) {
  return grid_unit == DispatchGridUnit::threads ? "threads" : "threadgroups";
}

} // namespace mlx_perf
