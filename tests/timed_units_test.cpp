// Component tests for timed_units: the uniform view over per-node and
// per-stage timelines, on synthetic measurements (no GPU dispatch).

#include "report_fixture.h"
#include "test_harness.h"
#include "timed_units.h"

#include "mlx/mlx.h"

#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

using report_fixture::SyntheticRun;
using mlx_perf::TimedUnit;

std::unordered_map<int, mlx::core::Stream> make_two_gpu_streams() {
  return {{0, mlx::core::Stream(0, mlx::core::Device::gpu)},
          {1, mlx::core::Stream(1, mlx::core::Device::gpu)}};
}

// --- positive ---

void should_view_each_node_as_a_unit_in_a_per_node_timeline() {
  SyntheticRun synthetic_run;
  report_fixture::populate_two_stream_run(synthetic_run);
  const std::vector<TimedUnit> timed_units =
      mlx_perf::collect_timed_units(synthetic_run.gpu_timeline);
  EXPECT_TRUE(timed_units.size() == 2);
  EXPECT_TRUE(timed_units[1].caption == "B<x> Add (stream 0)");
  EXPECT_TRUE(timed_units[1].cross_stream_producer_units == (std::vector<int>{0}));
  EXPECT_TRUE(std::string(mlx_perf::timed_unit_noun(synthetic_run.gpu_timeline)) == "node");
}

// The per-stage view must read the STAGE timings, never the zeroed nodes.
void should_view_each_stage_task_as_a_unit_in_a_per_stage_timeline() {
  SyntheticRun synthetic_run;
  report_fixture::populate_two_stage_run(synthetic_run);
  const std::vector<TimedUnit> timed_units =
      mlx_perf::collect_timed_units(synthetic_run.gpu_timeline);
  EXPECT_TRUE(timed_units.size() == 2);
  EXPECT_TRUE(timed_units[0].caption == "layer 0 (stream 1, 1 nodes)");
  EXPECT_TRUE(timed_units[1].breakdown->wait_on_dependency_microseconds > 0);
  const mlx_perf::LatencyBreakdown aggregate = mlx_perf::aggregate_breakdown(timed_units);
  EXPECT_NEAR(aggregate.wall_microseconds, 130.0 + 128.0, 1e-9);
  EXPECT_TRUE(std::string(mlx_perf::timed_unit_noun(synthetic_run.gpu_timeline)) == "stage");
}

void should_summarize_streams_from_units() {
  SyntheticRun synthetic_run;
  report_fixture::populate_two_stage_run(synthetic_run);
  const auto stream_timelines = mlx_perf::summarize_streams(
      mlx_perf::collect_timed_units(synthetic_run.gpu_timeline), make_two_gpu_streams());
  EXPECT_TRUE(stream_timelines.size() == 2);
  if (stream_timelines.size() == 2) {
    EXPECT_TRUE(stream_timelines[0].stream_index == 0);
    EXPECT_NEAR(stream_timelines[0].wall_microseconds, 140.0 - 12.0, 1e-9);
    EXPECT_TRUE(stream_timelines[1].dependency_wait_microseconds == 0);
  }
}

// --- edge / boundary ---

void should_return_no_units_and_no_streams_for_an_empty_timeline() {
  const mlx_perf::GpuTimeline empty_timeline;
  const std::vector<TimedUnit> timed_units = mlx_perf::collect_timed_units(empty_timeline);
  EXPECT_TRUE(timed_units.empty());
  EXPECT_TRUE(mlx_perf::summarize_streams(timed_units, {}).empty());
  EXPECT_TRUE(mlx_perf::aggregate_breakdown(timed_units).wall_microseconds == 0);
}

// --- negative ---

void should_throw_when_a_unit_stream_is_unknown() {
  SyntheticRun synthetic_run;
  report_fixture::populate_two_stream_run(synthetic_run);
  bool has_thrown = false;
  try {
    mlx_perf::summarize_streams(mlx_perf::collect_timed_units(synthetic_run.gpu_timeline), {});
  } catch (const std::out_of_range&) {
    has_thrown = true;
  }
  EXPECT_TRUE(has_thrown);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_view_each_node_as_a_unit_in_a_per_node_timeline),
      TEST_CASE(should_view_each_stage_task_as_a_unit_in_a_per_stage_timeline),
      TEST_CASE(should_summarize_streams_from_units),
      TEST_CASE(should_return_no_units_and_no_streams_for_an_empty_timeline),
      TEST_CASE(should_throw_when_a_unit_stream_is_unknown),
  });
}
