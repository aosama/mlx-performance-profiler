// The C entry point for procedure comparisons: adapts C callbacks to the
// ProcedureJob seam, measures, writes the report files and fills the C
// summary. Shared C API plumbing lives in c_api_support.

#include "mlx_perf/mlx_perf_c_api.h"

#include "c_api_support.h"
#include "procedure_measurement.h"
#include "procedure_report_files.h"

#include <stdexcept>
#include <string>

namespace {

using namespace mlx_perf;

constexpr const char* ENTRY_POINT_NAME = "mlx_perf_compare_procedures_to_report";

// A non-zero status from a callback becomes an exception here, after the C
// frame has returned, so no exception ever unwinds through caller code.
class CallbackProcedureJob final : public ProcedureJob {
 public:
  explicit CallbackProcedureJob(const mlx_perf_procedure& c_procedure) : c_procedure_(c_procedure) {}

  void prepare() override {
    if (c_procedure_.prepare != nullptr) {
      require_success(": prepare returned ", c_procedure_.prepare(c_procedure_.procedure_payload));
    }
  }

  void run() override { require_success(": run returned ", c_procedure_.run(c_procedure_.procedure_payload)); }

 private:
  // The failure text is spelled out whole at each call site, so the
  // troubleshooting page's quotes can be checked against this source.
  void require_success(const char* step_failure_text, int callback_status) const {
    if (callback_status != 0) {
      throw std::runtime_error("procedure " + std::string(c_procedure_.procedure_name) + step_failure_text +
                               std::to_string(callback_status));
    }
  }

  const mlx_perf_procedure& c_procedure_;
};

// Name rules (non-empty, no '/', distinct ignoring case) are checked by
// measure_procedure_pair; only what C can get wrong on top is checked here.
void require_procedures(const mlx_perf_procedure* baseline, const mlx_perf_procedure* candidate) {
  if (baseline == nullptr || candidate == nullptr) {
    throw std::invalid_argument("mlx_perf_compare_procedures_to_report: null procedure");
  }
  for (const mlx_perf_procedure* c_procedure : {baseline, candidate}) {
    if (c_procedure->run == nullptr) {
      throw std::invalid_argument("mlx_perf_compare_procedures_to_report: null run callback");
    }
    require_strings({c_procedure->procedure_name}, ENTRY_POINT_NAME);
  }
}

double median_of_rounds(const ProcedureMeasurement& procedure_measurement, int ProcedureRun::*count_field) {
  return summarize_samples(collect_round_counts(procedure_measurement, count_field)).median;
}

void fill_procedure_summary(const ProcedureMeasurement& procedure_measurement, int gpu_core_count,
                            mlx_perf_procedure_summary& procedure_summary) {
  procedure_summary.median_command_buffer_count =
      median_of_rounds(procedure_measurement, &ProcedureRun::command_buffer_count);
  procedure_summary.median_blocking_wait_count =
      median_of_rounds(procedure_measurement, &ProcedureRun::blocking_wait_count);
  procedure_summary.median_blocking_wait_us =
      summarize_samples(collect_round_values(procedure_measurement, &ProcedureRun::blocking_wait_microseconds))
          .median;
  procedure_summary.kernel_launch_count = static_cast<int>(procedure_measurement.observed_launches.size());
  procedure_summary.launches_unable_to_fill_gpu = gpu_core_count > 0 ? 0 : -1;
  for (const KernelDispatch& kernel_dispatch : procedure_measurement.observed_launches) {
    if (gpu_core_count > 0 && !can_fill_gpu(count_threadgroups(kernel_dispatch), gpu_core_count)) {
      ++procedure_summary.launches_unable_to_fill_gpu;
    }
  }
}

void compare_procedures_to_report(const mlx_perf_procedure* baseline, const mlx_perf_procedure* candidate,
                                  const char* report_title, const char* output_directory,
                                  const char* report_file_base, mlx_perf_procedure_comparison_summary* summary) {
  require_procedures(baseline, candidate);
  require_strings({report_title, output_directory, report_file_base}, ENTRY_POINT_NAME);
  CallbackProcedureJob baseline_job(*baseline);
  CallbackProcedureJob candidate_job(*candidate);
  const PairedProcedureMeasurement paired_measurement =
      measure_procedure_pair({baseline->procedure_name, baseline_job}, {candidate->procedure_name, candidate_job});
  const ProcedureComparisons procedure_comparisons = compare_procedure_pair(paired_measurement);
  if (!write_procedure_report_files({report_title, output_directory, report_file_base}, paired_measurement,
                                    procedure_comparisons)) {
    throw std::runtime_error("cannot write the procedure comparison into " + std::string(output_directory));
  }
  if (summary == nullptr) return;
  const int gpu_core_count = paired_measurement.gpu_device_facts.gpu_core_count;
  fill_procedure_summary(paired_measurement.baseline, gpu_core_count, summary->baseline);
  fill_procedure_summary(paired_measurement.candidate, gpu_core_count, summary->candidate);
  fill_metric_comparison(procedure_comparisons.duration, summary->duration);
  fill_metric_comparison(procedure_comparisons.gpu_busy, summary->gpu_busy);
  fill_metric_comparison(procedure_comparisons.gpu_idle, summary->gpu_idle);
}

} // namespace

extern "C" int mlx_perf_compare_procedures_to_report(const mlx_perf_procedure* baseline,
                                                     const mlx_perf_procedure* candidate,
                                                     const char* report_title, const char* output_directory,
                                                     const char* report_file_base,
                                                     mlx_perf_procedure_comparison_summary* summary) {
  clear_last_c_api_error();
  try {
    compare_procedures_to_report(baseline, candidate, report_title, output_directory, report_file_base, summary);
    return 0;
  } catch (...) {
    return record_failure_in_flight();
  }
}
