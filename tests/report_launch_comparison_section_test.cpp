// Component tests for report_launch_comparison_section: the comparison
// page's kernel launches side by side, from synthetic launches. No GPU.

#include "report_fixture.h"
#include "report_launch_comparison_section.h"
#include "test_harness.h"

#include <string>

namespace {

using namespace mlx_perf;

KernelDispatch launch_of_threadgroups(std::uint64_t threadgroup_count) {
  KernelDispatch kernel_dispatch;
  kernel_dispatch.grid = {threadgroup_count, 1, 1};
  kernel_dispatch.threads_per_threadgroup = {32, 1, 1};
  return kernel_dispatch;
}

std::size_t count_occurrences(const std::string& text, const std::string& needle) {
  std::size_t occurrence_count = 0;
  for (std::size_t found_at = text.find(needle); found_at != std::string::npos;
       found_at = text.find(needle, found_at + needle.size())) {
    ++occurrence_count;
  }
  return occurrence_count;
}

// One variant: a single node running `kernel_name` with one launch.
struct OneKernelRun {
  GraphNode kernel_node = report_fixture::make_graph_node("A", "CustomKernel", 0);
  RunTelemetry serial_telemetry;

  OneKernelRun(const std::string& kernel_name, std::uint64_t threadgroup_count) {
    kernel_node.kernel_name = kernel_name;
    serial_telemetry.attributed_nodes = {
        {&kernel_node, 40.0, 0, 64, {launch_of_threadgroups(threadgroup_count)}}};
    serial_telemetry.gpu_device_facts = {"Apple M5 Pro", 16};
  }
};

// --- positive ---

void should_pair_kernels_by_name_and_flag_the_variant_that_cannot_fill_the_gpu() {
  const OneKernelRun four_groups("custom_kernel_attention", 4);
  const OneKernelRun many_groups("custom_kernel_attention", 512);
  const std::string section_html = render_launch_comparison_section(
      "four_groups", four_groups.serial_telemetry, "many_groups", many_groups.serial_telemetry);
  EXPECT_TRUE(section_html.find("id=\"launches\"") != std::string::npos);
  EXPECT_TRUE(count_occurrences(section_html, "custom_kernel_attention") == 1);
  EXPECT_TRUE(section_html.find("⚠ 4 of 16") != std::string::npos);
  EXPECT_TRUE(section_html.find(">16 of 16<") != std::string::npos);
  EXPECT_TRUE(count_occurrences(section_html, "underfills-gpu") == 1);
}

// --- edge / boundary ---

void should_show_a_dash_for_a_kernel_only_one_variant_ran() {
  const OneKernelRun stock("Softmax", 256);
  const OneKernelRun custom("custom_kernel_softmax", 256);
  const std::string section_html = render_launch_comparison_section(
      "stock", stock.serial_telemetry, "custom", custom.serial_telemetry);
  EXPECT_TRUE(section_html.find("Softmax") != std::string::npos);
  EXPECT_TRUE(section_html.find("custom_kernel_softmax") != std::string::npos);
  EXPECT_TRUE(count_occurrences(section_html, ">—</td>") == 6);
}

void should_render_nothing_when_neither_variant_launched_a_kernel() {
  const RunTelemetry no_launches;
  EXPECT_TRUE(render_launch_comparison_section("a", no_launches, "b", no_launches).empty());
}

// --- negative ---

void should_escape_a_kernel_name_and_variant_names() {
  const OneKernelRun hostile("<script>x</script>", 4);
  const std::string section_html = render_launch_comparison_section(
      "<b>", hostile.serial_telemetry, "c", hostile.serial_telemetry);
  EXPECT_TRUE(section_html.find("<script>") == std::string::npos);
  EXPECT_TRUE(section_html.find("<b>") == std::string::npos);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_pair_kernels_by_name_and_flag_the_variant_that_cannot_fill_the_gpu),
      TEST_CASE(should_show_a_dash_for_a_kernel_only_one_variant_ran),
      TEST_CASE(should_render_nothing_when_neither_variant_launched_a_kernel),
      TEST_CASE(should_escape_a_kernel_name_and_variant_names),
  });
}
