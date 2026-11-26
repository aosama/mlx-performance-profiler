// Unit tests for wired_memory_sample: the figures derived from one sample.

#include "test_harness.h"
#include "wired_memory_fixture.h"
#include "wired_memory_sample.h"

namespace {

using namespace mlx_perf;
using wired_memory_fixture::GIB;
using wired_memory_fixture::sample_at;

// --- positive ---

void should_derive_holding_outside_and_headroom() {
  WiredMemorySample wired_memory_sample = sample_at(0, 3 * GIB, GIB);
  wired_memory_sample.mlx_cache_bytes = GIB / 2;
  wired_memory_sample.system_wired_bytes += GIB / 2;
  EXPECT_TRUE(mlx_held_bytes(wired_memory_sample) == GIB + GIB / 2);
  EXPECT_TRUE(wired_outside_mlx_bytes(wired_memory_sample) == static_cast<std::int64_t>(3 * GIB));
  EXPECT_TRUE(wire_headroom_bytes(wired_memory_sample) == static_cast<std::int64_t>(40 * GIB - 4 * GIB - GIB / 2));
}

void should_name_every_sample_point() {
  EXPECT_TRUE(wired_sample_point_name(WiredSamplePoint::run_start) == "run_start");
  EXPECT_TRUE(wired_sample_point_name(WiredSamplePoint::evaluation_boundary) == "evaluation_boundary");
  EXPECT_TRUE(wired_sample_point_name(WiredSamplePoint::interval) == "interval");
  EXPECT_TRUE(wired_sample_point_name(WiredSamplePoint::run_end) == "run_end");
  EXPECT_TRUE(wired_sample_point_name(WiredSamplePoint::after_run) == "after_run");
}

// --- edge / boundary ---

void should_go_negative_outside_mlx_when_macos_has_not_wired_mlx_memory_yet() {
  WiredMemorySample wired_memory_sample = sample_at(0, 0, 2 * GIB);
  wired_memory_sample.system_wired_bytes = GIB;
  EXPECT_TRUE(wired_outside_mlx_bytes(wired_memory_sample) == -static_cast<std::int64_t>(GIB));
}

void should_give_zero_headroom_when_the_limit_is_unknown() {
  EXPECT_TRUE(wire_headroom_bytes(sample_at(0, 3 * GIB, 0, 0, 0)) == 0);
}

// --- negative ---

void should_go_negative_headroom_over_the_limit() {
  EXPECT_TRUE(wire_headroom_bytes(sample_at(0, 41 * GIB, 0)) == -static_cast<std::int64_t>(GIB));
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_derive_holding_outside_and_headroom),
      TEST_CASE(should_name_every_sample_point),
      TEST_CASE(should_go_negative_outside_mlx_when_macos_has_not_wired_mlx_memory_yet),
      TEST_CASE(should_give_zero_headroom_when_the_limit_is_unknown),
      TEST_CASE(should_go_negative_headroom_over_the_limit),
  });
}
