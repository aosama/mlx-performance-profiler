#include "latency_slices.h"

namespace mlx_perf {

const SliceStyle& slice_style(LatencySlice latency_slice) {
  static const SliceStyle cpu_style{
      "slice-cpu", "host issue", "Scheduler Delay (host side)",
      "Host time inside async_eval: graph walk, kernel encode, commit."};
  static const SliceStyle queued_style{
      "slice-queued", "queued", "no Spark slice: stream queueing",
      "Issued, but earlier work on the same stream had not finished."};
  static const SliceStyle stalled_style{
      "slice-stalled", "stalled", "Shuffle Read fetchWaitTime",
      "Stream free, but an input produced on another stream had not arrived."};
  static const SliceStyle scheduling_style{
      "slice-scheduling", "GPU scheduling", "Scheduler Delay",
      "Committed and unblocked, but the GPU had not started the buffer."};
  static const SliceStyle compute_style{
      "slice-compute", "compute", "Executor Computing Time",
      "GPU execution of the timed unit's command buffer(s) (measured)."};
  static const SliceStyle result_style{
      "slice-result", "result", "Getting Result Time",
      "GPU finished → host noticed. Observation cost; never sets the class."};
  switch (latency_slice) {
    case LatencySlice::wait_on_cpu: return cpu_style;
    case LatencySlice::wait_on_gpu: return queued_style;
    case LatencySlice::wait_on_dependency: return stalled_style;
    case LatencySlice::scheduler_delay: return scheduling_style;
    case LatencySlice::compute: return compute_style;
    case LatencySlice::completion_latency: return result_style;
  }
  return compute_style;
}

LatencySlice slice_for_class(NodeClass node_class) {
  switch (node_class) {
    case NodeClass::compute: return LatencySlice::compute;
    case NodeClass::wait_on_cpu: return LatencySlice::wait_on_cpu;
    case NodeClass::wait_on_gpu: return LatencySlice::wait_on_gpu;
    case NodeClass::wait_on_dependency: return LatencySlice::wait_on_dependency;
    case NodeClass::scheduler_delay: return LatencySlice::scheduler_delay;
  }
  return LatencySlice::compute;
}

double slice_microseconds(
    const LatencyBreakdown& breakdown, LatencySlice latency_slice) {
  switch (latency_slice) {
    case LatencySlice::wait_on_cpu: return breakdown.wait_on_cpu_microseconds;
    case LatencySlice::wait_on_gpu: return breakdown.wait_on_gpu_microseconds;
    case LatencySlice::wait_on_dependency:
      return breakdown.wait_on_dependency_microseconds;
    case LatencySlice::scheduler_delay:
      return breakdown.scheduler_delay_microseconds;
    case LatencySlice::compute: return breakdown.compute_microseconds;
    case LatencySlice::completion_latency:
      return breakdown.completion_latency_microseconds;
  }
  return 0.0;
}

} // namespace mlx_perf
