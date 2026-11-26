// Unit tests for kernel_launch_summary: launches grouped by kernel, and a
// node's launches in one line. Synthetic launches, no GPU.

#include "kernel_launch_summary.h"
#include "report_fixture.h"
#include "test_harness.h"

#include <string>

namespace {

using namespace mlx_perf;

KernelDispatch launch_of_threadgroups(std::uint64_t threadgroup_count) {
  KernelDispatch kernel_dispatch;
  kernel_dispatch.grid_unit = DispatchGridUnit::threadgroups;
  kernel_dispatch.grid = {threadgroup_count, 1, 1};
  kernel_dispatch.threads_per_threadgroup = {32, 1, 1};
  return kernel_dispatch;
}

// Keeps the graph nodes alive for the telemetry that points into them.
struct LaunchedNodes {
  GraphNode first_softmax = report_fixture::make_graph_node("A", "Softmax", 0);
  GraphNode custom_kernel = report_fixture::make_graph_node("B", "CustomKernel", 0);
  GraphNode reshape = report_fixture::make_graph_node("C", "Reshape", 0);
  GraphNode second_softmax = report_fixture::make_graph_node("D", "Softmax", 0);
  RunTelemetry serial_telemetry;

  explicit LaunchedNodes(int gpu_core_count) {
    custom_kernel.kernel_name = "custom_kernel_four_groups";
    serial_telemetry.attributed_nodes = {
        {&first_softmax, 10.0, 0, 64, {launch_of_threadgroups(256)}},
        {&custom_kernel, 40.0, 0, 64, {launch_of_threadgroups(4)}},
        {&reshape, 1.0, 0, 64, {}},
        {&second_softmax, 10.0, 0, 64, {launch_of_threadgroups(8)}}};
    serial_telemetry.gpu_device_facts = {"Apple M5 Pro", gpu_core_count};
  }
};

// --- positive ---

void should_group_launches_by_kernel_in_first_run_order() {
  const LaunchedNodes launched_nodes(16);
  const auto kernel_summaries = summarize_kernel_launches(launched_nodes.serial_telemetry);
  EXPECT_TRUE(kernel_summaries.size() == 2);
  EXPECT_TRUE(kernel_summaries[0].kernel_key == "Softmax");
  EXPECT_TRUE(kernel_summaries[0].launch_count == 2);
  EXPECT_TRUE(kernel_summaries[0].fewest_threadgroups == 8 && kernel_summaries[0].most_threadgroups == 256);
  EXPECT_TRUE(kernel_summaries[0].lowest_core_coverage == 0.5);
  EXPECT_TRUE(kernel_summaries[0].has_launch_unable_to_fill_gpu);
}

void should_key_a_custom_kernel_by_its_name() {
  const LaunchedNodes launched_nodes(16);
  const auto kernel_summaries = summarize_kernel_launches(launched_nodes.serial_telemetry);
  EXPECT_TRUE(kernel_summaries[1].kernel_key == "custom_kernel_four_groups");
  EXPECT_TRUE(kernel_summaries[1].lowest_core_coverage == 0.25);
}

void should_describe_a_node_launch_in_one_line() {
  EXPECT_TRUE(describe_node_launches({launch_of_threadgroups(4)}, 16) == "4 threadgroups · 4 of 16 cores");
  EXPECT_TRUE(describe_node_launches({launch_of_threadgroups(8), launch_of_threadgroups(256)}, 16) ==
              "2 launches · 8–256 threadgroups · 8 of 16 cores");
}

// --- edge / boundary ---

void should_say_every_core_when_threadgroups_outnumber_cores() {
  EXPECT_TRUE(describe_node_launches({launch_of_threadgroups(512)}, 16) == "512 threadgroups · 16 of 16 cores");
  EXPECT_TRUE(describe_node_launches({launch_of_threadgroups(1)}, 16) == "1 threadgroup · 1 of 16 cores");
}

void should_leave_out_nodes_that_launched_nothing() {
  RunTelemetry views_only;
  GraphNode reshape = report_fixture::make_graph_node("C", "Reshape", 0);
  views_only.attributed_nodes = {{&reshape, 1.0, 0, 64, {}}};
  EXPECT_TRUE(summarize_kernel_launches(views_only).empty());
  EXPECT_TRUE(describe_node_launches({}, 16).empty());
}

// --- negative ---

void should_claim_no_coverage_when_the_core_count_is_unknown() {
  const LaunchedNodes launched_nodes(0);
  const auto kernel_summaries = summarize_kernel_launches(launched_nodes.serial_telemetry);
  EXPECT_TRUE(kernel_summaries[1].lowest_core_coverage < 0.0);
  EXPECT_TRUE(!kernel_summaries[1].has_launch_unable_to_fill_gpu);
  EXPECT_TRUE(describe_node_launches({launch_of_threadgroups(4)}, 0) == "4 threadgroups");
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_group_launches_by_kernel_in_first_run_order),
      TEST_CASE(should_key_a_custom_kernel_by_its_name),
      TEST_CASE(should_describe_a_node_launch_in_one_line),
      TEST_CASE(should_say_every_core_when_threadgroups_outnumber_cores),
      TEST_CASE(should_leave_out_nodes_that_launched_nothing),
      TEST_CASE(should_claim_no_coverage_when_the_core_count_is_unknown),
  });
}
