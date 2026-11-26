#include "procedure_comparison_record.h"

#include "comparison_record.h"
#include "kernel_launch_summary.h"
#include "measurement_document.h"
#include "wired_memory_record.h"

namespace mlx_perf {
namespace {

using Object = JsonValue::Object;
using Array = JsonValue::Array;

JsonValue run_record(const ProcedureRun& procedure_run) {
  return JsonValue(Object{
      {"duration_us", json_number(procedure_run.duration_microseconds)},
      {"gpu_busy_us", json_number(procedure_run.gpu_busy_microseconds)},
      {"gpu_idle_us", json_number(procedure_run.gpu_idle_microseconds)},
      {"command_buffer_count", json_number(procedure_run.command_buffer_count)},
      {"blocking_wait_count", json_number(procedure_run.blocking_wait_count)},
      {"blocking_wait_us", json_number(procedure_run.blocking_wait_microseconds)}});
}

// Unknown coverage (core count unknown) is null, never 0.
JsonValue launch_summary_record(const KernelLaunchSummary& function_summary) {
  const bool is_coverage_known = function_summary.lowest_core_coverage >= 0.0;
  return JsonValue(Object{
      {"metal_function_name", json_text(function_summary.kernel_key)},
      {"launch_count", json_number(function_summary.launch_count)},
      {"fewest_threadgroups", json_number(static_cast<double>(function_summary.fewest_threadgroups))},
      {"most_threadgroups", json_number(static_cast<double>(function_summary.most_threadgroups))},
      {"lowest_gpu_core_coverage",
       is_coverage_known ? json_number(function_summary.lowest_core_coverage) : JsonValue()},
      {"has_launch_unable_to_fill_gpu",
       is_coverage_known ? JsonValue(function_summary.has_launch_unable_to_fill_gpu) : JsonValue()}});
}

JsonValue procedure_record(const ProcedureMeasurement& procedure_measurement, int gpu_core_count) {
  Array round_records;
  for (const ProcedureRun& procedure_run : procedure_measurement.rounds) {
    round_records.push_back(run_record(procedure_run));
  }
  Array launch_records;
  for (const KernelLaunchSummary& function_summary :
       summarize_launches_by_function(procedure_measurement.observed_launches, gpu_core_count)) {
    launch_records.push_back(launch_summary_record(function_summary));
  }
  return JsonValue(Object{
      {"procedure_name", json_text(procedure_measurement.procedure_name)},
      {"rounds", JsonValue(std::move(round_records))},
      {"observed_run", run_record(procedure_measurement.observed_run)},
      {"kernel_launch_count", json_number(static_cast<double>(procedure_measurement.observed_launches.size()))},
      {"kernel_launches", JsonValue(std::move(launch_records))},
      {"wired_memory", build_wired_memory_record(procedure_measurement.observed_wired_memory)}});
}

JsonValue metric_record(const PairedComparison& paired_comparison, const PairedProcedureMeasurement& paired_measurement,
                 double ProcedureRun::*round_field) {
  return build_metric_record(paired_comparison, collect_round_values(paired_measurement.baseline, round_field),
                             collect_round_values(paired_measurement.candidate, round_field));
}

} // namespace

JsonValue build_procedure_comparison_record(const PairedProcedureMeasurement& paired_measurement,
                                            const ProcedureComparisons& procedure_comparisons) {
  const GpuDeviceFacts& gpu_device_facts = paired_measurement.gpu_device_facts;
  const int gpu_core_count = gpu_device_facts.gpu_core_count;
  return JsonValue(Object{
      {"schema_version", json_number(MEASUREMENT_SCHEMA_VERSION)},
      {"kind", json_text("procedure_comparison")},
      {"gpu", JsonValue(Object{
          {"name", gpu_device_facts.gpu_name.empty() ? JsonValue() : json_text(gpu_device_facts.gpu_name)},
          {"core_count", gpu_core_count > 0 ? json_number(gpu_core_count) : JsonValue()}})},
      {"baseline", procedure_record(paired_measurement.baseline, gpu_core_count)},
      {"candidate", procedure_record(paired_measurement.candidate, gpu_core_count)},
      {"metrics", JsonValue(Object{
          {"duration_us", metric_record(procedure_comparisons.duration, paired_measurement,
                                 &ProcedureRun::duration_microseconds)},
          {"gpu_busy_us", metric_record(procedure_comparisons.gpu_busy, paired_measurement,
                                 &ProcedureRun::gpu_busy_microseconds)},
          {"gpu_idle_us", metric_record(procedure_comparisons.gpu_idle, paired_measurement,
                                 &ProcedureRun::gpu_idle_microseconds)}})}});
}

} // namespace mlx_perf
