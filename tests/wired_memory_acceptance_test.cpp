// Acceptance test for #45 on the Metal GPU: an engine-like procedure drops a
// 1 GiB table part-way through its run and keeps evaluating, as an engine does
// when a prompt ends. The procedure comparison must show, for that run, the
// release in MLX's holding, macOS un-wiring it later, and the samples in
// between whose "wired outside MLX" counted MLX's released memory as someone
// else's, on the page and in the record. Spans the sampler, the submission
// recorder's evaluation boundaries, the analysis, the record and the page, so
// it is named for the behaviour, not one module.

#include "json_value.h"
#include "procedure_measurement.h"
#include "procedure_report_files.h"
#include "test_harness.h"

#include "mlx/mlx.h"

#include <fstream>
#include <iterator>
#include <optional>
#include <string>

namespace {

using namespace mlx::core;
using mlx_perf::ProcedureJob;

constexpr int TABLE_ELEMENT_COUNT = 1 << 28;  // 1 GiB of float32
constexpr double GIB = 1024.0 * 1024.0 * 1024.0;

class TableProcedure final : public ProcedureJob {
 public:
  explicit TableProcedure(bool is_dropping_table) : is_dropping_table_(is_dropping_table) {}

  void prepare() override {
    lookup_table_ = zeros({TABLE_ELEMENT_COUNT}, float32) + 1.0f;
    eval(*lookup_table_);
  }

  // Reads the table, then (when dropping) releases it as a finished prompt's
  // cache would be, and goes on evaluating small steps, reading each back.
  void run() override {
    array running_value = sum(slice(*lookup_table_, {0}, {1 << 20}));
    eval(running_value);
    if (is_dropping_table_) {
      // MLX returns a buffer to its cache only once nothing refers to it;
      // the synchronize lets that happen before the cache is cleared.
      lookup_table_.reset();
      synchronize();
      clear_cache();
    }
    for (int step_index = 0; step_index < 8; ++step_index) {
      running_value = running_value * 1.0001f + 1.0f;
      running_value.item<float>();
    }
  }

 private:
  bool is_dropping_table_;
  std::optional<array> lookup_table_;
};

std::string read_file(const std::string& file_path) {
  std::ifstream input_stream(file_path);
  return std::string(std::istreambuf_iterator<char>(input_stream), {});
}

void should_show_the_dropped_table_unwired_late_and_the_samples_it_inflated() {
  TableProcedure keeping_table(false);
  TableProcedure dropping_table(true);
  const auto paired_measurement =
      mlx_perf::measure_procedure_pair({"keep_table", keeping_table}, {"drop_table", dropping_table});
  EXPECT_TRUE(mlx_perf::write_procedure_report_files({"Dropping a 1 GiB table mid-run", ".", "acceptance_wired_memory"},
                                                     paired_measurement,
                                                     mlx_perf::compare_procedure_pair(paired_measurement)));
  const mlx_perf::JsonValue record = mlx_perf::parse_json_file("acceptance_wired_memory.json");
  const mlx_perf::JsonValue& dropping_wired = record.at("candidate").at("wired_memory");
  EXPECT_TRUE(dropping_wired.at("peak_mlx_held_bytes").as_number() >= GIB);
  EXPECT_TRUE(dropping_wired.at("wire_limit_bytes").as_number() > dropping_wired.at("peak_system_wired_bytes").as_number());
  const mlx_perf::JsonValue::Array& release_events = dropping_wired.at("release_events").as_array();
  EXPECT_TRUE(!release_events.empty());
  if (!release_events.empty()) {
    const mlx_perf::JsonValue& table_release = release_events.front();
    EXPECT_TRUE(table_release.at("released_bytes").as_number() >= 0.9 * GIB);
    EXPECT_TRUE(!table_release.at("release_lag_us").is_null());
    // Wherever macOS took longer than a couple of samples, those samples
    // must be marked, with up to about the table's size inflating them.
    if (!table_release.at("release_lag_us").is_null() && table_release.at("release_lag_us").as_number() > 3000.0) {
      EXPECT_TRUE(!dropping_wired.at("release_lag_inflated_sample_indices").as_array().empty());
      EXPECT_TRUE(dropping_wired.at("largest_release_lag_inflation_bytes").as_number() >= 0.5 * GIB);
    }
  }
  EXPECT_TRUE(record.at("baseline").at("wired_memory").at("release_events").as_array().empty());
  const std::string comparison_page = read_file("acceptance_wired_memory.html");
  EXPECT_TRUE(comparison_page.find("id=\"wired-memory-candidate\"") != std::string::npos);
  EXPECT_TRUE(comparison_page.find("un-wired after") != std::string::npos);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_show_the_dropped_table_unwired_late_and_the_samples_it_inflated),
  });
}
