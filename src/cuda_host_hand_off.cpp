#include "cuda_host_hand_off.h"

namespace mlx_perf {

bool is_graph_launch(CUpti_CallbackId callback_id) {
  return callback_id == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch ||
      callback_id == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz;
}

bool is_host_wait_or_completion_check(CUpti_CallbackId callback_id) {
  switch (callback_id) {
    case CUPTI_DRIVER_TRACE_CBID_cuEventSynchronize:
    case CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize:
    case CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize:
    case CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuEventQuery:
      return true;
    default:
      return false;
  }
}

bool counts_as_host_hand_off(CUpti_CallbackId callback_id, CUresult returned_result) {
  if (!is_host_wait_or_completion_check(callback_id)) {
    return false;
  }
  if (callback_id == CUPTI_DRIVER_TRACE_CBID_cuEventQuery) {
    return returned_result == CUDA_SUCCESS;
  }
  return true;
}

} // namespace mlx_perf
