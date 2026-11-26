// Component tests for wired_memory_record: a run's wired memory as JSON,
// written and read back, from synthetic samples. No GPU.

#include "json_value.h"
#include "json_writer.h"
#include "test_harness.h"
#include "wired_memory_fixture.h"
#include "wired_memory_record.h"

namespace {

using namespace mlx_perf;
using wired_memory_fixture::GIB;
using wired_memory_fixture::sample_at;

JsonValue written_and_parsed(const std::vector<WiredMemorySample>& wired_memory_samples) {
  return parse_json(write_json(build_wired_memory_record(wired_memory_samples)));
}

// --- positive ---

void should_carry_the_summary_the_release_and_every_sample() {
  const JsonValue record = written_and_parsed(wired_memory_fixture::release_then_unwire());
  EXPECT_TRUE(record.at("wire_limit_bytes").as_number() == static_cast<double>(40 * GIB));
  EXPECT_TRUE(record.at("sample_count").as_number() == 8);
  EXPECT_TRUE(record.at("peak_system_wired_bytes").as_number() == static_cast<double>(4 * GIB));
  EXPECT_TRUE(record.at("peak_wired_outside_mlx_bytes").as_number() == static_cast<double>(4 * GIB));
  EXPECT_TRUE(record.at("lowest_wired_outside_mlx_bytes").as_number() == static_cast<double>(3 * GIB));
  EXPECT_TRUE(record.at("lowest_wire_headroom_bytes").as_number() == static_cast<double>(36 * GIB));
  EXPECT_TRUE(!record.at("has_crossed_wire_limit").as_bool());
  const JsonValue& release_event = record.at("release_events").as_array().at(0);
  EXPECT_TRUE(release_event.at("released_bytes").as_number() == static_cast<double>(GIB));
  EXPECT_TRUE(release_event.at("unwired_sample_index").as_number() == 6);
  EXPECT_NEAR(release_event.at("release_lag_us").as_number(), 30000.0, 1e-3);
  EXPECT_NEAR(record.at("median_release_lag_us").as_number(), 30000.0, 1e-3);
  EXPECT_TRUE(record.at("release_lag_inflated_sample_indices").as_array().size() == 3);
  EXPECT_TRUE(record.at("samples").as_array().size() == 8);
}

void should_time_samples_from_the_first_and_derive_their_figures() {
  const JsonValue record = written_and_parsed(wired_memory_fixture::release_then_unwire());
  const JsonValue& fourth_sample = record.at("samples").as_array().at(3);
  EXPECT_NEAR(fourth_sample.at("time_us").as_number(), 10000.0, 1e-3);
  EXPECT_TRUE(fourth_sample.at("sample_point").as_string() == "interval");
  EXPECT_TRUE(fourth_sample.at("mlx_active_bytes").as_number() == 0);
  EXPECT_TRUE(fourth_sample.at("wired_outside_mlx_bytes").as_number() == static_cast<double>(4 * GIB));
  EXPECT_TRUE(fourth_sample.at("wire_headroom_bytes").as_number() == static_cast<double>(36 * GIB));
}

// --- edge / boundary ---

void should_write_null_for_no_samples() {
  EXPECT_TRUE(written_and_parsed({}).is_null());
}

void should_write_null_lags_for_an_unresolved_release() {
  const JsonValue record = written_and_parsed({sample_at(0, GIB, GIB), sample_at(1, GIB, 0, GIB)});
  EXPECT_TRUE(record.at("release_events").as_array().at(0).at("unwired_sample_index").is_null());
  EXPECT_TRUE(record.at("median_release_lag_us").is_null());
  EXPECT_TRUE(record.at("largest_release_lag_us").is_null());
}

// --- negative ---

void should_write_null_limit_figures_when_the_limit_is_unknown() {
  const JsonValue record = written_and_parsed({sample_at(0, 3 * GIB, 0, 0, 0)});
  EXPECT_TRUE(record.at("wire_limit_bytes").is_null());
  EXPECT_TRUE(record.at("lowest_wire_headroom_bytes").is_null());
  EXPECT_TRUE(record.at("has_crossed_wire_limit").is_null());
  EXPECT_TRUE(record.at("samples").as_array().at(0).at("wire_headroom_bytes").is_null());
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_carry_the_summary_the_release_and_every_sample),
      TEST_CASE(should_time_samples_from_the_first_and_derive_their_figures),
      TEST_CASE(should_write_null_for_no_samples),
      TEST_CASE(should_write_null_lags_for_an_unresolved_release),
      TEST_CASE(should_write_null_limit_figures_when_the_limit_is_unknown),
  });
}
