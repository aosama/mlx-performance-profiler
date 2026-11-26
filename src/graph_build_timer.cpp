#include "graph_build_timer.h"

#include <chrono>

namespace mlx_perf {
namespace {

using Clock = std::chrono::steady_clock;

constexpr const char* UNLABELLED_PASS_NAME = "unlabelled";

double microseconds_between(Clock::time_point started_at, Clock::time_point finished_at) {
  return std::chrono::duration<double, std::micro>(finished_at - started_at).count();
}

} // namespace

TimedGraphFactory::TimedGraphFactory(const GraphFactory& timed_factory)
    : timed_factory_(timed_factory) {}

void TimedGraphFactory::begin_pass(const std::string& pass_name) {
  build_passes_.push_back({pass_name, {}});
}

// A call that throws is not recorded: it built no graph, and the exception
// carries the job's failure to the caller unchanged.
std::vector<mlx::core::array> TimedGraphFactory::build_fresh_graph() const {
  const Clock::time_point started_at = Clock::now();
  std::vector<mlx::core::array> output_arrays = timed_factory_.build_fresh_graph();
  record_call(microseconds_between(started_at, Clock::now()));
  return output_arrays;
}

StagedGraph TimedGraphFactory::build_fresh_staged_graph() const {
  const Clock::time_point started_at = Clock::now();
  StagedGraph staged_graph = timed_factory_.build_fresh_staged_graph();
  record_call(microseconds_between(started_at, Clock::now()));
  return staged_graph;
}

void TimedGraphFactory::record_call(double call_microseconds) const {
  if (build_passes_.empty()) {
    build_passes_.push_back({UNLABELLED_PASS_NAME, {}});
  }
  build_passes_.back().call_microseconds.push_back(call_microseconds);
}

} // namespace mlx_perf
