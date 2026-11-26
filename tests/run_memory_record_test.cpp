// Component tests for run_memory_record: one run's memory figures and a set
// of rounds as JSON, written and read back. No GPU.

#include "json_value.h"
#include "json_writer.h"
#include "run_memory_record.h"
#include "test_harness.h"

namespace {

using namespace mlx_perf;

RunMemoryFigures readable_figures() {
  RunMemoryFigures memory_figures;
  memory_figures.peak_mlx_active_bytes = 3000;
  memory_figures.peak_mlx_active_added_bytes = 1000;
  memory_figures.mlx_held_at_end_bytes = 2000;
  memory_figures.mlx_held_change_bytes = -500;
  memory_figures.system_wired_at_end_bytes = 4000;
  memory_figures.wire_limit_bytes = 8000;
  memory_figures.compressed_bytes = 16;
  memory_figures.swap_used_change_bytes = -32;
  memory_figures.are_pressure_counters_readable = true;
  return memory_figures;
}

JsonValue written_and_parsed(const JsonValue& memory_record) { return parse_json(write_json(memory_record)); }

// --- positive ---

void should_carry_every_figure_and_both_flags() {
  const JsonValue memory_record = written_and_parsed(build_run_memory_record(readable_figures()));
  EXPECT_TRUE(memory_record.at("peak_mlx_active_added_bytes").as_number() == 1000);
  EXPECT_TRUE(memory_record.at("mlx_held_change_bytes").as_number() == -500);
  EXPECT_TRUE(memory_record.at("wire_limit_bytes").as_number() == 8000);
  EXPECT_TRUE(memory_record.at("swap_used_change_bytes").as_number() == -32);
  EXPECT_TRUE(!memory_record.at("has_crossed_wire_limit").as_bool());
  EXPECT_TRUE(memory_record.at("has_compressed_memory").as_bool());
}

void should_count_the_rounds_under_pressure() {
  RunMemoryFigures crossing_round = readable_figures();
  crossing_round.wire_requests_over_limit_count = 3;
  const JsonValue rounds_record =
      written_and_parsed(build_rounds_memory_record({readable_figures(), crossing_round}));
  EXPECT_TRUE(rounds_record.at("rounds_crossing_wire_limit").as_number() == 1);
  EXPECT_TRUE(rounds_record.at("rounds_with_compression").as_number() == 2);
  EXPECT_TRUE(rounds_record.at("rounds").as_array().size() == 2);
}

// --- edge / boundary ---

void should_write_an_empty_set_of_rounds() {
  const JsonValue rounds_record = written_and_parsed(build_rounds_memory_record({}));
  EXPECT_TRUE(rounds_record.at("rounds_crossing_wire_limit").as_number() == 0);
  EXPECT_TRUE(rounds_record.at("rounds").as_array().empty());
}

// --- negative ---

void should_write_null_where_the_kernel_gave_no_evidence() {
  RunMemoryFigures unreadable_figures = readable_figures();
  unreadable_figures.are_pressure_counters_readable = false;
  unreadable_figures.wire_limit_bytes = 0;
  const JsonValue memory_record = written_and_parsed(build_run_memory_record(unreadable_figures));
  EXPECT_TRUE(memory_record.at("wire_limit_bytes").is_null());
  EXPECT_TRUE(memory_record.at("compressed_bytes").is_null());
  EXPECT_TRUE(memory_record.at("has_crossed_wire_limit").is_null());
  EXPECT_TRUE(memory_record.at("has_compressed_memory").is_null());
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_carry_every_figure_and_both_flags),
      TEST_CASE(should_count_the_rounds_under_pressure),
      TEST_CASE(should_write_an_empty_set_of_rounds),
      TEST_CASE(should_write_null_where_the_kernel_gave_no_evidence),
  });
}
