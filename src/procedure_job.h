#pragma once

// Procedure job: the second kind of job, beside GraphFactory. A graph job
// hands the profiler one fresh graph to evaluate; a procedure job is code
// that evaluates by itself, any number of times, and may change state between
// evaluations, such as an inference engine reading a prompt layer by layer.
// It is measured as it is served: the profiler times run() whole and observes
// what MLX issues inside it.

#include <string>

namespace mlx_perf {

class ProcedureJob {
 public:
  virtual ~ProcedureJob() = default;
  // Restores the state run() starts from. Never timed. Evaluate what it sets
  // up before returning, so that work is not left to the timed run.
  virtual void prepare() {}
  // The procedure itself: timed whole, including any GPU work it leaves in
  // flight. Called on the thread that measures; MLX work it issues on that
  // thread is attributed to it.
  virtual void run() = 0;
};

struct ComparedProcedure {
  // Shown on the page and in the record: non-empty, no '/', and distinct
  // from the other procedure's ignoring case.
  std::string procedure_name;
  ProcedureJob& procedure_job;
};

} // namespace mlx_perf
