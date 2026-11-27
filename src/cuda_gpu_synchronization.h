#pragma once

// CUDA GPU synchronization: one concern only — wait until the GPU has
// finished every piece of work submitted so far, from any thread.
//
// Why the whole context and not the streams the work went to: a procedure may
// evaluate on an owner thread that has exited by the time its record is read,
// and MLX's streams (and so their CUDA streams) can go with that thread, so a
// stored stream handle may dangle. MLX uses the device's primary context on
// every thread, so waiting on it covers every MLX stream.

namespace mlx_perf {

// Makes the primary context current for a calling thread that never touched
// the GPU itself. Does nothing when there is no CUDA device.
void wait_for_every_gpu_submission();

} // namespace mlx_perf
