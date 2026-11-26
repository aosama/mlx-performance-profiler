// Integration tests for procedure_measurement on the Metal GPU: two real
// procedures — one that evaluates after every step, one that evaluates once —
// measured whole, with their hand-offs counted and their launches observed.

#include "procedure_measurement.h"
#include "test_harness.h"

#include "mlx/mlx.h"

#include <stdexcept>
#include <string>
#include <thread>

namespace {

using namespace mlx_perf;
using mlx::core::array;

// `step_count` small elementwise steps over a fresh input; evaluates after
// every step when `is_evaluating_every_step`, else once at the end.
class SteppedProcedure final : public ProcedureJob {
 public:
  SteppedProcedure(int step_count, bool is_evaluating_every_step)
      : step_count_(step_count), is_evaluating_every_step_(is_evaluating_every_step) {}

  void prepare() override {
    ++prepare_count;
    step_input_ = std::make_unique<array>(mlx::core::ones({256, 256}));
    mlx::core::eval(*step_input_);
  }

  void run() override {
    ++run_count;
    array running_value = *step_input_;
    for (int step_index = 0; step_index < step_count_; ++step_index) {
      running_value = running_value * 1.0001f + 1.0f;
      if (is_evaluating_every_step_) mlx::core::eval(running_value);
    }
    // Nothing left to evaluate when every step was: the chatty run hands off
    // `step_count` times, the batched run once.
    mlx::core::eval(running_value);
  }

  int prepare_count = 0;
  int run_count = 0;

 private:
  int step_count_;
  bool is_evaluating_every_step_;
  std::unique_ptr<array> step_input_;
};

// #44: run() sends the steps to an owner thread for MLX and waits for it,
// as an engine with its own MLX thread is driven.
class OwnerThreadProcedure final : public ProcedureJob {
 public:
  void prepare() override { steps_.prepare(); }
  void run() override {
    std::thread owner_thread([this]() { steps_.run(); });
    owner_thread.join();
  }

 private:
  SteppedProcedure steps_{6, true};
};

class HostOnlyProcedure final : public ProcedureJob {
 public:
  void run() override { ++run_count; }
  int run_count = 0;
};

class FailingProcedure final : public ProcedureJob {
 public:
  void run() override { throw std::runtime_error("procedure failed on purpose"); }
};

bool is_refused(const std::string& baseline_name, const std::string& candidate_name) {
  HostOnlyProcedure baseline_job;
  HostOnlyProcedure candidate_job;
  try {
    measure_procedure_pair({baseline_name, baseline_job}, {candidate_name, candidate_job});
  } catch (const std::invalid_argument&) {
    return baseline_job.run_count == 0 && candidate_job.run_count == 0;
  }
  return false;
}

// --- positive ---

void should_count_the_hand_offs_of_each_procedure_per_run() {
  SteppedProcedure chatty_job(8, true);
  SteppedProcedure batched_job(8, false);
  const PairedProcedureMeasurement paired_measurement =
      measure_procedure_pair({"chatty", chatty_job}, {"batched", batched_job});
  EXPECT_TRUE(paired_measurement.baseline.rounds.size() == PROCEDURE_ROUND_COUNT);
  EXPECT_TRUE(paired_measurement.candidate.rounds.size() == PROCEDURE_ROUND_COUNT);
  for (const ProcedureRun& chatty_run : paired_measurement.baseline.rounds) {
    EXPECT_TRUE(chatty_run.blocking_wait_count >= 8);
    EXPECT_TRUE(chatty_run.command_buffer_count >= 8);
  }
  for (const ProcedureRun& batched_run : paired_measurement.candidate.rounds) {
    EXPECT_TRUE(batched_run.blocking_wait_count >= 1 && batched_run.blocking_wait_count <= 2);
    EXPECT_TRUE(batched_run.gpu_busy_microseconds > 0.0);
    EXPECT_TRUE(batched_run.gpu_busy_microseconds <= batched_run.duration_microseconds + 1e-6);
    EXPECT_NEAR(batched_run.gpu_idle_microseconds,
                batched_run.duration_microseconds - batched_run.gpu_busy_microseconds, 1e-6);
  }
  EXPECT_TRUE(paired_measurement.baseline.observed_launches.size() >=
              paired_measurement.candidate.observed_launches.size());
  EXPECT_TRUE(!paired_measurement.candidate.observed_launches.empty());
  EXPECT_TRUE(!paired_measurement.candidate.observed_launches.front().metal_function_name.empty());
  EXPECT_TRUE(compare_procedure_pair(paired_measurement).duration.round_count == PROCEDURE_ROUND_COUNT);
  // The observed run carries a wired memory track, from its start to its end.
  const std::vector<WiredMemorySample>& chatty_wired_memory = paired_measurement.baseline.observed_wired_memory;
  EXPECT_TRUE(chatty_wired_memory.size() >= 2);
  EXPECT_TRUE(!chatty_wired_memory.empty() && chatty_wired_memory.front().sample_point == WiredSamplePoint::run_start);
  int boundary_sample_count = 0;
  for (const WiredMemorySample& wired_memory_sample : chatty_wired_memory) {
    boundary_sample_count += wired_memory_sample.sample_point == WiredSamplePoint::evaluation_boundary ? 1 : 0;
  }
  EXPECT_TRUE(boundary_sample_count >= 8);
}

void should_prepare_before_every_run_of_both_procedures() {
  SteppedProcedure baseline_job(2, false);
  SteppedProcedure candidate_job(2, true);
  measure_procedure_pair({"baseline", baseline_job}, {"candidate", candidate_job});
  // One warm-up, one observed run, then one run per round.
  EXPECT_TRUE(baseline_job.prepare_count == 2 + PROCEDURE_ROUND_COUNT);
  EXPECT_TRUE(baseline_job.run_count == baseline_job.prepare_count);
  EXPECT_TRUE(candidate_job.run_count == 2 + PROCEDURE_ROUND_COUNT);
}

void should_observe_a_procedure_whose_mlx_work_runs_on_another_thread() {
  OwnerThreadProcedure owner_thread_job;
  SteppedProcedure calling_thread_job(6, true);
  const PairedProcedureMeasurement paired_measurement =
      measure_procedure_pair({"calling_thread", calling_thread_job}, {"owner_thread", owner_thread_job});
  for (const ProcedureRun& owner_thread_run : paired_measurement.candidate.rounds) {
    EXPECT_TRUE(owner_thread_run.command_buffer_count >= 6);
    EXPECT_TRUE(owner_thread_run.blocking_wait_count >= 6);
    EXPECT_TRUE(owner_thread_run.gpu_busy_microseconds > 0.0);
    EXPECT_TRUE(owner_thread_run.gpu_idle_microseconds < owner_thread_run.duration_microseconds);
  }
  EXPECT_TRUE(!paired_measurement.candidate.observed_launches.empty());
}

// --- edge / boundary ---

void should_record_no_gpu_time_for_procedures_without_mlx_work() {
  HostOnlyProcedure baseline_job;
  HostOnlyProcedure candidate_job;
  const PairedProcedureMeasurement paired_measurement =
      measure_procedure_pair({"empty_a", baseline_job}, {"empty_b", candidate_job});
  for (const ProcedureRun& empty_run : paired_measurement.baseline.rounds) {
    EXPECT_TRUE(empty_run.command_buffer_count == 0);
    EXPECT_TRUE(empty_run.blocking_wait_count == 0);
    EXPECT_TRUE(empty_run.gpu_busy_microseconds == 0.0);
    EXPECT_TRUE(empty_run.duration_microseconds >= 0.0);
  }
  EXPECT_TRUE(paired_measurement.baseline.observed_launches.empty());
}

// --- negative ---

void should_refuse_unusable_names_before_running_anything() {
  EXPECT_TRUE(is_refused("", "candidate"));
  EXPECT_TRUE(is_refused("base/line", "candidate"));
  EXPECT_TRUE(is_refused("Read", "read"));
  EXPECT_TRUE(!is_refused("read_a", "read_b"));
}

void should_end_the_measurement_when_a_procedure_throws() {
  HostOnlyProcedure baseline_job;
  FailingProcedure failing_job;
  bool has_thrown = false;
  try {
    measure_procedure_pair({"baseline", baseline_job}, {"failing", failing_job});
  } catch (const std::runtime_error& procedure_error) {
    has_thrown = std::string(procedure_error.what()) == "procedure failed on purpose";
  }
  EXPECT_TRUE(has_thrown);
  EXPECT_TRUE(baseline_job.run_count == 1);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_count_the_hand_offs_of_each_procedure_per_run),
      TEST_CASE(should_prepare_before_every_run_of_both_procedures),
      TEST_CASE(should_observe_a_procedure_whose_mlx_work_runs_on_another_thread),
      TEST_CASE(should_record_no_gpu_time_for_procedures_without_mlx_work),
      TEST_CASE(should_refuse_unusable_names_before_running_anything),
      TEST_CASE(should_end_the_measurement_when_a_procedure_throws),
  });
}
