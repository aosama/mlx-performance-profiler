// Acceptance test for #35 on the Metal GPU: one custom kernel doing the same
// work, launched as 4 threadgroups and as hundreds. The comparison shows both
// launch shapes side by side and flags the 4-threadgroup launch as unable to
// fill the GPU, the "why" behind a kernel that is slower for the same work.
// Spans the dispatch recorder, the serial pass, the measurement file and the
// comparison page, so it is named for the behaviour, not one module.

#include "gpu_device_facts.h"
#include "json_value.h"
#include "paired_measurement.h"
#include "report_bundle.h"
#include "custom_kernel_fixture.h"
#include "test_harness.h"

#include "mlx/fast.h"
#include "mlx/mlx.h"

#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>

namespace {

using namespace mlx::core;
using mlx_perf::ComparedJob;
using mlx_perf::GraphFactory;

constexpr int ELEMENT_COUNT = 1 << 22;
constexpr int THREADS_PER_THREADGROUP = 32;

// Each thread doubles every grid-stride element, so any launch size does the
// same total work: only the number of threadgroups differs between variants.
const auto scale_elements = custom_kernel_fixture::make_grid_stride_doubling_kernel("grid_stride_double");

class GridStrideFactory final : public GraphFactory {
 public:
  GridStrideFactory(const array& input_values, int threadgroup_count)
      : input_values_(input_values), threadgroup_count_(threadgroup_count) {}
  std::vector<array> build_fresh_graph() const override {
    return scale_elements({input_values_}, {input_values_.shape()}, {float32},
                          {threadgroup_count_ * THREADS_PER_THREADGROUP, 1, 1},
                          {THREADS_PER_THREADGROUP, 1, 1}, {}, std::nullopt, false, {});
  }

 private:
  const array& input_values_;
  int threadgroup_count_;
};

std::string read_file(const std::string& file_path) {
  std::ifstream input_stream(file_path);
  return std::string(std::istreambuf_iterator<char>(input_stream), {});
}

const mlx_perf::JsonValue& first_launch(const mlx_perf::JsonValue& measurement_document) {
  for (const mlx_perf::JsonValue& node_record : measurement_document.at("nodes").as_array()) {
    if (node_record.at("op_name").as_string() == "CustomKernel") {
      return node_record.at("kernel_dispatches").as_array().at(0);
    }
  }
  return measurement_document; // reached only when the kernel is missing: the checks then fail
}

void should_show_both_launch_shapes_and_flag_the_one_that_cannot_fill_the_gpu() {
  const array input_values = ones({ELEMENT_COUNT});
  eval(input_values);
  const GridStrideFactory four_threadgroups(input_values, 4);
  const GridStrideFactory many_threadgroups(input_values, 512);
  const Stream job_stream = default_stream(Device::gpu);
  const std::unordered_map<int, Stream> stream_index_to_stream{{job_stream.index, job_stream}};
  const auto paired_measurement = mlx_perf::measure_pair(
      ComparedJob{"four_groups", four_threadgroups}, ComparedJob{"many_groups", many_threadgroups},
      stream_index_to_stream);
  EXPECT_TRUE(mlx_perf::write_report_bundle({"Grid-stride doubling", ".", "acceptance_launch_occupancy"},
                                            paired_measurement));

  const int gpu_core_count = mlx_perf::read_gpu_device_facts().gpu_core_count;
  EXPECT_TRUE(gpu_core_count > 4);
  const auto four_groups_launch =
      first_launch(mlx_perf::parse_json_file("acceptance_launch_occupancy_four_groups.json"));
  const auto many_groups_launch =
      first_launch(mlx_perf::parse_json_file("acceptance_launch_occupancy_many_groups.json"));
  EXPECT_TRUE(four_groups_launch.at("threadgroup_count").as_number() == 4);
  EXPECT_TRUE(!four_groups_launch.at("can_fill_gpu").as_bool());
  EXPECT_TRUE(many_groups_launch.at("threadgroup_count").as_number() == 512);
  EXPECT_TRUE(many_groups_launch.at("can_fill_gpu").as_bool());
  EXPECT_TRUE(four_groups_launch.at("kernel_function_name").as_string().find("grid_stride_double") !=
              std::string::npos);

  // One page shows both shapes side by side and flags the 4-threadgroup one.
  const std::string comparison_page = read_file("acceptance_launch_occupancy.html");
  const std::size_t launches_at = comparison_page.find("id=\"launches\"");
  EXPECT_TRUE(launches_at != std::string::npos);
  const std::string launches_section =
      comparison_page.substr(launches_at, comparison_page.find("</section>", launches_at) - launches_at);
  const std::string cores_text = " of " + std::to_string(gpu_core_count);
  EXPECT_TRUE(launches_section.find("⚠ 4" + cores_text) != std::string::npos);
  EXPECT_TRUE(launches_section.find(">" + std::to_string(gpu_core_count) + cores_text + "<") !=
              std::string::npos);
  EXPECT_TRUE(launches_section.find(">512<") != std::string::npos);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_show_both_launch_shapes_and_flag_the_one_that_cannot_fill_the_gpu),
  });
}
