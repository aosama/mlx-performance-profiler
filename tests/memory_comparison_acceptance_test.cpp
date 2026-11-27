// Acceptance test for #46 on the Metal GPU: two versions of one step, the
// second building a 256 MiB scratch working set and keeping 32 MiB more
// after every run, as a cache that is never trimmed would. The procedure
// comparison must judge memory on the same rounds as time and call both the
// larger working set and what the run leaves behind beyond the spread, on the
// page and in the record, with each round's memory figures and its memory
// pressure counts. Spans the memory meter, the procedure measurement, the
// record and the page, so it is named for the behaviour, not one module.

#include "json_value.h"
#include "procedure_measurement.h"
#include "procedure_report_files.h"
#include "backend_expectations.h"
#include "test_harness.h"

#include "mlx/mlx.h"

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

using namespace mlx::core;
using mlx_perf::ProcedureJob;

constexpr double MIB = 1024.0 * 1024.0;

class CachingStep final : public ProcedureJob {
 public:
  explicit CachingStep(bool is_hoarding) : is_hoarding_(is_hoarding) {}

  void run() override {
    array step_result = sum(matmul(ones({512, 512}), ones({512, 512})));
    if (is_hoarding_) {
      // A 256 MiB scratch buffer the step uses and drops, and 32 MiB it keeps.
      array scratch = zeros({64 * 1024 * 1024}, float32) + 1.0f;
      step_result = step_result + sum(scratch);
      kept_blocks_.push_back(zeros({8 * 1024 * 1024}, float32) + 2.0f);
      eval(step_result, kept_blocks_.back());
    } else {
      eval(step_result);
    }
  }

 private:
  bool is_hoarding_;
  std::vector<array> kept_blocks_;
};

std::string read_file(const std::string& file_path) {
  std::ifstream input_stream(file_path);
  return std::string(std::istreambuf_iterator<char>(input_stream), {});
}

void should_judge_the_larger_working_set_and_what_the_run_keeps_beyond_the_spread() {
  CachingStep lean_step(false);
  CachingStep hoarding_step(true);
  const auto paired_measurement =
      mlx_perf::measure_procedure_pair({"lean", lean_step}, {"hoarding", hoarding_step});
  EXPECT_TRUE(mlx_perf::write_procedure_report_files({"A step that keeps a cache", ".",
                                                      "acceptance_memory_comparison"},
                                                     paired_measurement,
                                                     mlx_perf::compare_procedure_pair(paired_measurement)));
  const mlx_perf::JsonValue record = mlx_perf::parse_json_file("acceptance_memory_comparison.json");
  const mlx_perf::JsonValue& metrics = record.at("metrics");
  EXPECT_TRUE(metrics.at("peak_mlx_active_added_bytes").at("verdict").as_string() == "candidate_slower");
  EXPECT_TRUE(metrics.at("peak_mlx_active_added_bytes").at("median_difference_bytes").as_number() >= 200 * MIB);
  EXPECT_TRUE(metrics.at("mlx_held_change_bytes").at("verdict").as_string() == "candidate_slower");
  EXPECT_TRUE(metrics.at("mlx_held_change_bytes").at("median_difference_bytes").as_number() >= 24 * MIB);
  const mlx_perf::JsonValue& first_round_memory = record.at("candidate").at("rounds").as_array().at(0).at("memory");
  EXPECT_TRUE(first_round_memory.at("peak_mlx_active_added_bytes").as_number() >= 256 * MIB);
  EXPECT_TRUE(backend_expectations::HAS_SYSTEM_WIRED_MEMORY ? first_round_memory.at("wire_limit_bytes").as_number() > 0
                                                         : first_round_memory.at("wire_limit_bytes").is_null());
  EXPECT_TRUE(record.at("candidate").at("rounds_crossing_wire_limit").as_number() >= 0);
  const std::string comparison_page = read_file("acceptance_memory_comparison.html");
  EXPECT_TRUE(comparison_page.find("<h2>Memory per run</h2>") != std::string::npos);
  EXPECT_TRUE(comparison_page.find("more, beyond the spread") != std::string::npos);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_judge_the_larger_working_set_and_what_the_run_keeps_beyond_the_spread),
  });
}
