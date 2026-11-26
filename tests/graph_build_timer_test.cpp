// Unit tests for graph_build_timer: calls are counted, timed and filed under
// the pass that made them; the wrapped factory's failures pass through.

#include "graph_build_timer.h"
#include "test_harness.h"

#include "mlx/array.h"

#include <stdexcept>
#include <string>

namespace {

using mlx_perf::GraphFactory;
using mlx_perf::StagedGraph;
using mlx_perf::TimedGraphFactory;

// Builds a scalar graph and counts how often it was asked to. Never
// evaluated, so no GPU is needed.
class CountingFactory final : public GraphFactory {
 public:
  std::vector<mlx::core::array> build_fresh_graph() const override {
    ++build_call_count;
    return {mlx::core::array(1.0f)};
  }
  mutable int build_call_count = 0;
};

class FailingFactory final : public GraphFactory {
 public:
  std::vector<mlx::core::array> build_fresh_graph() const override {
    throw std::runtime_error("job could not build its graph");
  }
};

// --- positive ---

void should_file_each_call_under_the_pass_that_made_it() {
  const CountingFactory counting_factory;
  TimedGraphFactory timed_factory(counting_factory);
  timed_factory.begin_pass("warm_up");
  timed_factory.build_fresh_graph();
  timed_factory.begin_pass("job_wall");
  for (int sample_index = 0; sample_index < 3; ++sample_index) {
    timed_factory.build_fresh_graph();
  }
  const auto& build_passes = timed_factory.build_passes();
  EXPECT_TRUE(build_passes.size() == 2);
  EXPECT_TRUE(build_passes[0].pass_name == "warm_up");
  EXPECT_TRUE(build_passes[0].call_microseconds.size() == 1);
  EXPECT_TRUE(build_passes[1].pass_name == "job_wall");
  EXPECT_TRUE(build_passes[1].call_microseconds.size() == 3);
  EXPECT_TRUE(counting_factory.build_call_count == 4);
  for (const double call_microseconds : build_passes[1].call_microseconds) {
    EXPECT_TRUE(call_microseconds >= 0.0);
  }
}

void should_time_staged_builds_through_the_wrapped_factory() {
  const CountingFactory counting_factory;
  TimedGraphFactory timed_factory(counting_factory);
  timed_factory.begin_pass("timeline");
  const StagedGraph staged_graph = timed_factory.build_fresh_staged_graph();
  EXPECT_TRUE(staged_graph.output_arrays.size() == 1);
  EXPECT_TRUE(timed_factory.build_passes()[0].call_microseconds.size() == 1);
  EXPECT_TRUE(counting_factory.build_call_count == 1);
}

// --- edge / boundary ---

void should_file_calls_before_any_pass_as_unlabelled() {
  const CountingFactory counting_factory;
  TimedGraphFactory timed_factory(counting_factory);
  timed_factory.build_fresh_graph();
  EXPECT_TRUE(timed_factory.build_passes().size() == 1);
  EXPECT_TRUE(timed_factory.build_passes()[0].pass_name == "unlabelled");
}

void should_keep_a_pass_that_made_no_calls() {
  const CountingFactory counting_factory;
  TimedGraphFactory timed_factory(counting_factory);
  timed_factory.begin_pass("serial");
  EXPECT_TRUE(timed_factory.build_passes().size() == 1);
  EXPECT_TRUE(timed_factory.build_passes()[0].call_microseconds.empty());
}

// --- negative ---

void should_pass_a_failing_build_through_without_recording_it() {
  const FailingFactory failing_factory;
  TimedGraphFactory timed_factory(failing_factory);
  timed_factory.begin_pass("warm_up");
  bool has_thrown = false;
  try {
    timed_factory.build_fresh_graph();
  } catch (const std::runtime_error& build_failure) {
    has_thrown = std::string(build_failure.what()) == "job could not build its graph";
  }
  EXPECT_TRUE(has_thrown);
  EXPECT_TRUE(timed_factory.build_passes()[0].call_microseconds.empty());
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_file_each_call_under_the_pass_that_made_it),
      TEST_CASE(should_time_staged_builds_through_the_wrapped_factory),
      TEST_CASE(should_file_calls_before_any_pass_as_unlabelled),
      TEST_CASE(should_keep_a_pass_that_made_no_calls),
      TEST_CASE(should_pass_a_failing_build_through_without_recording_it),
  });
}
