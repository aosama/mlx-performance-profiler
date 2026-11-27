// Integration tests for cupti_session: one CUPTI subscription routing real
// MLX driver calls to observers, and the GPU spans of watched graph launches
// reported on the host clock. Checked for presence, ordering and clock
// alignment, never exact timings.

#include "cupti_session.h"
#include "gpu_host_clock.h"
#include "test_harness.h"

#include "mlx/mlx.h"

#include <cstdint>
#include <vector>

namespace {

using namespace mlx::core;
using mlx_perf::DriverCall;
using mlx_perf::GpuActivitySpan;

// CUPTI converts GPU timestamps onto the clock the session hands it; the
// conversion's error is far below this.
constexpr double CLOCK_SLACK_NANOSECONDS = 50'000.0;

// Watches every graph launch it is told about, as the probe and the
// submission recorder do.
class GraphLaunchCollector final : public mlx_perf::DriverCallObserver {
 public:
  void on_driver_call(const DriverCall& driver_call) override {
    if (driver_call.is_entering && is_collecting) {
      launch_correlation_ids.push_back(driver_call.correlation_id);
      mlx_perf::watch_gpu_activity(driver_call.correlation_id);
    }
  }
  bool is_collecting = false;
  std::vector<std::uint32_t> launch_correlation_ids;
};

GraphLaunchCollector& graph_launch_collector() {
  static GraphLaunchCollector collector;
  static const bool is_observing =
      mlx_perf::observe_driver_call(CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch, &collector);
  (void)is_observing;
  return collector;
}

std::uint64_t now_nanoseconds() {
  return mlx_perf::current_session_nanoseconds();
}

// One blocking evaluation, its graph launches collected.
std::vector<std::uint32_t> collect_one_evaluation() {
  GraphLaunchCollector& collector = graph_launch_collector();
  array gpu_output = matmul(ones({256, 256}), ones({256, 256}));
  eval(gpu_output.inputs());
  collector.launch_correlation_ids.clear();
  collector.is_collecting = true;
  eval(gpu_output);
  collector.is_collecting = false;
  return collector.launch_correlation_ids;
}

// --- positive ---

void should_start_once_and_give_the_same_answer() {
  EXPECT_TRUE(mlx_perf::start_cupti_session());
  EXPECT_TRUE(mlx_perf::start_cupti_session());
}

void should_report_a_watched_launch_inside_its_host_window() {
  graph_launch_collector();
  const std::uint64_t before_nanoseconds = now_nanoseconds();
  const std::vector<std::uint32_t> correlation_ids = collect_one_evaluation();
  const std::uint64_t after_nanoseconds = now_nanoseconds();
  EXPECT_TRUE(!correlation_ids.empty());
  mlx_perf::flush_gpu_activity();
  for (const std::uint32_t correlation_id : correlation_ids) {
    const GpuActivitySpan activity_span = mlx_perf::take_gpu_activity_span(correlation_id);
    EXPECT_TRUE(activity_span.is_measured);
    EXPECT_TRUE(activity_span.activity_record_count >= 1);
    EXPECT_TRUE(activity_span.finished_nanoseconds >= activity_span.started_nanoseconds);
    EXPECT_TRUE(static_cast<double>(activity_span.started_nanoseconds) >=
                static_cast<double>(before_nanoseconds) - CLOCK_SLACK_NANOSECONDS);
    EXPECT_TRUE(static_cast<double>(activity_span.finished_nanoseconds) <=
                static_cast<double>(after_nanoseconds) + CLOCK_SLACK_NANOSECONDS);
  }
}

// --- edge / boundary ---

// The probe and the submission recorder can watch one launch: the span
// stays until the last of them has taken it.
void should_keep_a_span_until_every_watcher_took_it() {
  graph_launch_collector();
  const std::vector<std::uint32_t> correlation_ids = collect_one_evaluation();
  EXPECT_TRUE(!correlation_ids.empty());
  const std::uint32_t correlation_id = correlation_ids.front();
  mlx_perf::watch_gpu_activity(correlation_id);  // a second watcher
  mlx_perf::flush_gpu_activity();
  EXPECT_TRUE(mlx_perf::take_gpu_activity_span(correlation_id).is_measured);
  EXPECT_TRUE(mlx_perf::take_gpu_activity_span(correlation_id).is_measured);
  EXPECT_TRUE(!mlx_perf::take_gpu_activity_span(correlation_id).is_measured);
}

void should_flush_with_nothing_watched() {
  mlx_perf::flush_gpu_activity();
  mlx_perf::flush_gpu_activity();
  EXPECT_TRUE(mlx_perf::start_cupti_session());
}

// --- negative ---

void should_not_measure_an_unwatched_correlation_id() {
  constexpr std::uint32_t NEVER_WATCHED_CORRELATION_ID = 0xFFFFFFF0u;
  mlx_perf::flush_gpu_activity();
  EXPECT_TRUE(!mlx_perf::take_gpu_activity_span(NEVER_WATCHED_CORRELATION_ID).is_measured);
}

void should_refuse_a_null_observer() {
  EXPECT_TRUE(!mlx_perf::observe_driver_call(CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch, nullptr));
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_start_once_and_give_the_same_answer),
      TEST_CASE(should_report_a_watched_launch_inside_its_host_window),
      TEST_CASE(should_keep_a_span_until_every_watcher_took_it),
      TEST_CASE(should_flush_with_nothing_watched),
      TEST_CASE(should_not_measure_an_unwatched_correlation_id),
      TEST_CASE(should_refuse_a_null_observer),
  });
}
