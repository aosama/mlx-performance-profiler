#pragma once

// CUDA host hand-off: one concern only — decide, from a driver call's id and
// result, whether the call was a GPU submission or a moment the host learned
// the GPU had finished (a hand-off). Pure rules, no CUPTI state.
//
// Why a completion check can be a hand-off: MLX checks an evaluation's event
// before it waits on it (array::is_available, cudaEventQuery), and a fast GPU
// has often finished by then, so the host learns the work is done without
// blocking. That is the same hand-off as a wait that blocked. A check that
// found the work still running is only a poll and is not one.

#include <cuda.h>
#include <cupti.h>

namespace mlx_perf {

// MLX submits each stream's work as one CUDA graph launch.
bool is_graph_launch(CUpti_CallbackId callback_id);

// The waits MLX makes (an eval's event wait, synchronize, a device-wide
// wait) and the event check it makes before waiting.
bool is_host_wait_or_completion_check(CUpti_CallbackId callback_id);

// On the call's return: true for every wait, and for a completion check
// that found the work done.
bool counts_as_host_hand_off(CUpti_CallbackId callback_id, CUresult returned_result);

} // namespace mlx_perf
