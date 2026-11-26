// Acceptance test for #38 on the Metal GPU: one 32-step loop written two ways
// — reading a value back on the host after every step, as a token loop does,
// and evaluating once at the end. Neither is one graph, so only a procedure
// comparison can measure them. It must call the batched loop faster, show
// that the difference is GPU idle time (the host and GPU waiting on each
// other), and count the hand-offs that cause it. Spans the submission and
// dispatch recorders, the procedure measurement, the page and the record, so
// it is named for the behaviour, not one module.

#include "json_value.h"
#include "procedure_measurement.h"
#include "procedure_report_files.h"
#include "test_harness.h"

#include "mlx/mlx.h"

#include <fstream>
#include <iterator>
#include <memory>
#include <string>

namespace {

using namespace mlx::core;
using mlx_perf::ProcedureJob;

constexpr int LOOP_STEP_COUNT = 32;

class MatmulLoop final : public ProcedureJob {
 public:
  explicit MatmulLoop(bool is_reading_every_step) : is_reading_every_step_(is_reading_every_step) {}

  void prepare() override {
    loop_weights_ = std::make_unique<array>(random::normal({256, 256}) * 0.05f);
    loop_state_ = std::make_unique<array>(random::normal({1, 256}));
    eval(*loop_weights_, *loop_state_);
  }

  void run() override {
    array running_state = *loop_state_;
    for (int step_index = 0; step_index < LOOP_STEP_COUNT; ++step_index) {
      running_state = tanh(matmul(running_state, *loop_weights_));
      // What a sampler does with each token: bring a value to the host,
      // which blocks until the GPU has finished this step.
      if (is_reading_every_step_) sum(running_state).item<float>();
    }
    eval(running_state);
  }

 private:
  bool is_reading_every_step_;
  std::unique_ptr<array> loop_weights_;
  std::unique_ptr<array> loop_state_;
};

std::string read_file(const std::string& file_path) {
  std::ifstream input_stream(file_path);
  return std::string(std::istreambuf_iterator<char>(input_stream), {});
}

void should_call_the_batched_loop_faster_and_show_the_hand_offs_that_cost_the_other() {
  MatmulLoop reading_every_step(true);
  MatmulLoop evaluating_once(false);
  const auto paired_measurement =
      mlx_perf::measure_procedure_pair({"read_every_step", reading_every_step}, {"eval_once", evaluating_once});
  const auto procedure_comparisons = mlx_perf::compare_procedure_pair(paired_measurement);
  EXPECT_TRUE(mlx_perf::write_procedure_report_files({"A 32-step loop: read every step vs eval once", ".",
                                                      "acceptance_procedure_handoffs"},
                                                     paired_measurement, procedure_comparisons));

  const mlx_perf::JsonValue record = mlx_perf::parse_json_file("acceptance_procedure_handoffs.json");
  EXPECT_TRUE(record.at("kind").as_string() == "procedure_comparison");
  EXPECT_TRUE(record.at("metrics").at("duration_us").at("verdict").as_string() == "candidate_faster");
  EXPECT_TRUE(record.at("metrics").at("gpu_idle_us").at("verdict").as_string() == "candidate_faster");
  // Most of what the batched loop saves is idle time, not GPU work.
  const double duration_saved = -record.at("metrics").at("duration_us").at("median_difference_us").as_number();
  const double idle_saved = -record.at("metrics").at("gpu_idle_us").at("median_difference_us").as_number();
  EXPECT_TRUE(idle_saved > 0.5 * duration_saved);
  for (const mlx_perf::JsonValue& round_record : record.at("baseline").at("rounds").as_array()) {
    EXPECT_TRUE(round_record.at("blocking_wait_count").as_number() >= LOOP_STEP_COUNT);
  }
  for (const mlx_perf::JsonValue& round_record : record.at("candidate").at("rounds").as_array()) {
    EXPECT_TRUE(round_record.at("blocking_wait_count").as_number() <= 2);
  }

  const std::string comparison_page = read_file("acceptance_procedure_handoffs.html");
  EXPECT_TRUE(comparison_page.find("eval_once is") != std::string::npos);
  EXPECT_TRUE(comparison_page.find("id=\"handoffs\"") != std::string::npos);
  EXPECT_TRUE(comparison_page.find("id=\"launches\"") != std::string::npos);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_call_the_batched_loop_faster_and_show_the_hand_offs_that_cost_the_other),
  });
}
