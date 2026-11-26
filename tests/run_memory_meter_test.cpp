// Tests for run_memory_meter: the pure judgements and helpers on synthetic
// figures (unit), and the meter around real MLX work on the Metal GPU
// (integration). Memory figures are checked for direction and plausibility,
// never exact sizes.

#include "run_memory_meter.h"
#include "test_harness.h"

#include "mlx/mlx.h"

#include <optional>
#include <vector>

namespace {

using namespace mlx_perf;

constexpr std::uint64_t MIB = 1ull << 20;

RunMemoryFigures figures_over_limit_by(std::int64_t over_limit_bytes, std::uint64_t refused_count) {
  RunMemoryFigures memory_figures;
  memory_figures.wire_limit_bytes = 40ull << 30;
  memory_figures.system_wired_at_end_bytes = static_cast<std::uint64_t>((40ll << 30) + over_limit_bytes);
  memory_figures.wire_requests_over_limit_count = refused_count;
  memory_figures.are_pressure_counters_readable = true;
  return memory_figures;
}

// --- positive ---

void should_measure_a_runs_own_working_set_and_what_it_kept() {
  mlx::core::eval(mlx::core::ones({16}));  // MLX initialised outside the run
  std::optional<mlx::core::array> kept_table;
  RunMemoryMeter run_memory_meter;
  run_memory_meter.start();
  kept_table = mlx::core::zeros({static_cast<int>(256 * MIB / sizeof(float))}, mlx::core::float32);
  mlx::core::eval(*kept_table);
  mlx::core::synchronize();
  const RunMemoryFigures memory_figures = run_memory_meter.finish();
  // Less whatever was already active as the run began (a few bytes here).
  EXPECT_TRUE(memory_figures.peak_mlx_active_added_bytes >= 255 * MIB);
  EXPECT_TRUE(memory_figures.peak_mlx_active_bytes >= memory_figures.peak_mlx_active_added_bytes);
  EXPECT_TRUE(memory_figures.mlx_held_change_bytes >= static_cast<std::int64_t>(256 * MIB));
  EXPECT_TRUE(memory_figures.system_wired_at_end_bytes > 0);
  EXPECT_TRUE(memory_figures.wire_limit_bytes > memory_figures.system_wired_at_end_bytes);
  EXPECT_TRUE(memory_figures.are_pressure_counters_readable);
}

void should_see_a_negative_change_when_a_run_frees_what_it_found() {
  std::optional<mlx::core::array> found_table =
      mlx::core::zeros({static_cast<int>(128 * MIB / sizeof(float))}, mlx::core::float32);
  mlx::core::eval(*found_table);
  mlx::core::synchronize();
  RunMemoryMeter run_memory_meter;
  run_memory_meter.start();
  found_table.reset();
  mlx::core::synchronize();
  mlx::core::clear_cache();
  const RunMemoryFigures memory_figures = run_memory_meter.finish();
  EXPECT_TRUE(memory_figures.mlx_held_change_bytes <= -static_cast<std::int64_t>(64 * MIB));
  EXPECT_TRUE(memory_figures.peak_mlx_active_added_bytes == 0);
}

void should_count_runs_that_crossed_the_limit_or_compressed() {
  RunMemoryFigures compressing_run = figures_over_limit_by(-(1ll << 30), 0);
  compressing_run.compressed_bytes = 4096;
  const std::vector<RunMemoryFigures> run_memory_figures = {figures_over_limit_by(-(1ll << 30), 2), compressing_run,
                                                            figures_over_limit_by(-(1ll << 30), 0)};
  EXPECT_TRUE(count_runs_crossing_wire_limit(run_memory_figures) == 1);
  EXPECT_TRUE(count_runs_with_compression(run_memory_figures) == 1);
  EXPECT_TRUE((collect_run_bytes(run_memory_figures, &RunMemoryFigures::compressed_bytes) ==
               std::vector<double>{0, 4096, 0}));
}

// --- edge / boundary ---

void should_call_a_run_ending_over_the_limit_a_crossing_even_without_a_refusal() {
  EXPECT_TRUE(has_crossed_wire_limit(figures_over_limit_by(1, 0)));
  EXPECT_TRUE(!has_crossed_wire_limit(figures_over_limit_by(0, 0)));
}

void should_read_almost_nothing_for_a_run_that_does_nothing() {
  RunMemoryMeter run_memory_meter;
  run_memory_meter.start();
  const RunMemoryFigures memory_figures = run_memory_meter.finish();
  EXPECT_TRUE(memory_figures.peak_mlx_active_added_bytes == 0);
  EXPECT_TRUE(memory_figures.mlx_held_change_bytes == 0);
  EXPECT_TRUE(memory_figures.wire_requests_over_limit_count == 0);
}

// --- negative ---

void should_not_call_a_crossing_when_the_limit_is_unknown_and_nothing_was_refused() {
  RunMemoryFigures memory_figures = figures_over_limit_by(1ll << 30, 0);
  memory_figures.wire_limit_bytes = 0;
  EXPECT_TRUE(!has_crossed_wire_limit(memory_figures));
  EXPECT_TRUE(!has_compressed_memory(memory_figures));
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_measure_a_runs_own_working_set_and_what_it_kept),
      TEST_CASE(should_see_a_negative_change_when_a_run_frees_what_it_found),
      TEST_CASE(should_count_runs_that_crossed_the_limit_or_compressed),
      TEST_CASE(should_call_a_run_ending_over_the_limit_a_crossing_even_without_a_refusal),
      TEST_CASE(should_read_almost_nothing_for_a_run_that_does_nothing),
      TEST_CASE(should_not_call_a_crossing_when_the_limit_is_unknown_and_nothing_was_refused),
  });
}
