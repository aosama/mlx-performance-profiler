#pragma once

// Evaluation boundary listener: one concern only — the seam through which the
// submission recorder tells another module that a blocking wait for the GPU
// has just returned: the moment an engine reads its state after evaluating.
// The recorder knows when; the listener decides what to do then (for example,
// take a wired memory sample), so neither module depends on the other's job.

namespace mlx_perf {

class EvaluationBoundaryListener {
 public:
  virtual ~EvaluationBoundaryListener() = default;
  // Called on the thread whose wait returned, possibly a worker thread, and,
  // for a whole-process recording, while that recording's lock is held: keep
  // it short, and never evaluate MLX work from it.
  virtual void on_evaluation_boundary() = 0;
};

} // namespace mlx_perf
