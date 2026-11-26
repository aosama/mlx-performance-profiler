// Unit tests for latency_slices: the single source of slice vocabulary.

#include "latency_slices.h"
#include "test_harness.h"

#include <set>
#include <string>

namespace {

using namespace mlx_perf;

// --- positive ---

void should_give_every_slice_a_distinct_css_class_and_label() {
  std::set<std::string> css_classes;
  std::set<std::string> short_labels;
  for (const LatencySlice latency_slice : CAUSAL_SLICE_ORDER) {
    css_classes.insert(slice_style(latency_slice).css_class);
    short_labels.insert(slice_style(latency_slice).short_label);
  }
  EXPECT_TRUE(css_classes.size() == CAUSAL_SLICE_ORDER.size());
  EXPECT_TRUE(short_labels.size() == CAUSAL_SLICE_ORDER.size());
}

void should_read_each_slice_from_its_own_breakdown_field() {
  LatencyBreakdown breakdown;
  breakdown.wait_on_cpu_microseconds = 1;
  breakdown.wait_on_gpu_microseconds = 2;
  breakdown.wait_on_dependency_microseconds = 3;
  breakdown.scheduler_delay_microseconds = 4;
  breakdown.compute_microseconds = 5;
  breakdown.completion_latency_microseconds = 6;
  double expected_microseconds = 1;
  for (const LatencySlice latency_slice : CAUSAL_SLICE_ORDER) {
    EXPECT_TRUE(slice_microseconds(breakdown, latency_slice) ==
                expected_microseconds);
    expected_microseconds += 1;
  }
}

// --- edge / boundary ---

void should_map_every_class_to_the_slice_it_is_named_after() {
  EXPECT_TRUE(slice_for_class(NodeClass::compute) == LatencySlice::compute);
  EXPECT_TRUE(slice_for_class(NodeClass::wait_on_cpu) == LatencySlice::wait_on_cpu);
  EXPECT_TRUE(slice_for_class(NodeClass::wait_on_gpu) == LatencySlice::wait_on_gpu);
  EXPECT_TRUE(slice_for_class(NodeClass::wait_on_dependency) ==
              LatencySlice::wait_on_dependency);
  EXPECT_TRUE(slice_for_class(NodeClass::scheduler_delay) ==
              LatencySlice::scheduler_delay);
}

void should_keep_the_slices_summing_to_wall_in_causal_order() {
  LatencyBreakdown breakdown;
  breakdown.wait_on_cpu_microseconds = 1.5;
  breakdown.compute_microseconds = 2.5;
  breakdown.wall_microseconds = 4.0;
  double summed_microseconds = 0.0;
  for (const LatencySlice latency_slice : CAUSAL_SLICE_ORDER) {
    summed_microseconds += slice_microseconds(breakdown, latency_slice);
  }
  EXPECT_NEAR(summed_microseconds, breakdown.wall_microseconds, 1e-12);
}

// --- negative ---

void should_fall_back_to_compute_style_for_an_out_of_range_slice() {
  const auto corrupt_slice = static_cast<LatencySlice>(99);
  EXPECT_TRUE(std::string(slice_style(corrupt_slice).css_class) == "slice-compute");
  EXPECT_TRUE(slice_microseconds(LatencyBreakdown{}, corrupt_slice) == 0.0);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_give_every_slice_a_distinct_css_class_and_label),
      TEST_CASE(should_read_each_slice_from_its_own_breakdown_field),
      TEST_CASE(should_map_every_class_to_the_slice_it_is_named_after),
      TEST_CASE(should_keep_the_slices_summing_to_wall_in_causal_order),
      TEST_CASE(should_fall_back_to_compute_style_for_an_out_of_range_slice),
  });
}
