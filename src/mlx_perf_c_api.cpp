#include "mlx_perf/mlx_perf_c_api.h"

#include "graph_factory.h"
#include "graph_build_timer.h"
#include "graph_model.h"
#include "measurement_file.h"
#include "paired_measurement.h"
#include "report_bundle.h"
#include "measurement_session.h"
#include "mlx_c_interop.h"
#include "report_writer.h"

#include <exception>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace {

using namespace mlx_perf;

thread_local std::string last_error_message;

// Adapts the C callback to the GraphFactory seam the measurement uses: a new
// MLX-C vector per call, filled by the caller, read, then freed here.
class CallbackGraphFactory final : public GraphFactory {
 public:
  CallbackGraphFactory(mlx_perf_build_graph_fn build_graph, void* builder_payload)
      : build_graph_(build_graph), builder_payload_(builder_payload) {}

  std::vector<mlx::core::array> build_fresh_graph() const override {
    mlx_vector_array fresh_outputs = mlx_vector_array_new();
    const int callback_status = build_graph_(fresh_outputs, builder_payload_);
    if (callback_status != 0) {
      mlx_vector_array_free(fresh_outputs);
      throw std::runtime_error("build_graph callback returned " +
                               std::to_string(callback_status));
    }
    std::vector<mlx::core::array> output_nodes;
    try {
      output_nodes = read_mlx_c_vector(fresh_outputs);
    } catch (...) {
      mlx_vector_array_free(fresh_outputs);
      throw;
    }
    mlx_vector_array_free(fresh_outputs);
    if (output_nodes.empty()) {
      throw std::runtime_error("build_graph callback produced no outputs");
    }
    return output_nodes;
  }

 private:
  mlx_perf_build_graph_fn build_graph_;
  void* builder_payload_;
};

void fill_summary(const VariantMeasurement& variant_measurement,
                  mlx_perf_job_summary& summary) {
  const JobTotals& job_totals = variant_measurement.job_totals;
  const GpuTimeline& gpu_timeline = variant_measurement.gpu_timeline;
  summary.job_wall_median_us = job_totals.job_wall_microseconds;
  summary.job_wall_min_us = job_totals.job_wall_min_microseconds;
  summary.job_wall_max_us = job_totals.job_wall_max_microseconds;
  summary.timeline_wall_us = gpu_timeline.timeline_wall_microseconds;
  summary.sum_of_stages_us = job_totals.sum_of_stages_microseconds;
  summary.measured_gpu_execution_us = gpu_timeline.measured_gpu_compute_microseconds;
  summary.cross_stream_stall_us = gpu_timeline.cross_stream_stall_microseconds;
  summary.executed_node_count = static_cast<int>(gpu_timeline.nodes.size());
  summary.gpu_measured_node_count = gpu_timeline.gpu_measured_node_count;
  summary.stream_count = static_cast<int>(gpu_timeline.streams.size());
}

// The callback and every string the entry point reads must be present; the
// message names the entry point so a caller in another language can tell
// which call it got wrong.
void require_arguments(mlx_perf_build_graph_fn build_graph,
                       std::initializer_list<const char*> required_strings,
                       const char* entry_point_name) {
  bool has_null_argument = build_graph == nullptr;
  for (const char* required_string : required_strings) {
    has_null_argument |= required_string == nullptr;
  }
  if (has_null_argument) {
    throw std::invalid_argument(std::string(entry_point_name) + ": null argument");
  }
}

// Both entry points measure the same way. The stream-discovery build is
// timed like every other build and reported first, so a caller sees all
// eleven calls into its callback.
VariantMeasurement measure_callback_job(mlx_perf_build_graph_fn build_graph,
                                        void* builder_payload, const char* variant_name) {
  const CallbackGraphFactory graph_factory(build_graph, builder_payload);
  TimedGraphFactory discovery_factory(graph_factory);
  discovery_factory.begin_pass("stream_discovery");
  // Streams are read from the job's own primitives on a throwaway graph.
  const auto stream_index_to_stream =
      collect_streams(capture_graph(discovery_factory.build_fresh_graph()));
  VariantMeasurement variant_measurement =
      measure_variant(variant_name, graph_factory, stream_index_to_stream);
  variant_measurement.graph_build_passes.insert(
      variant_measurement.graph_build_passes.begin(),
      discovery_factory.build_passes().begin(), discovery_factory.build_passes().end());
  return variant_measurement;
}

void write_measurements_or_throw(const VariantMeasurement& variant_measurement,
                                 const char* output_directory, const char* file_base) {
  if (write_measurement_file(variant_measurement, output_directory, file_base).empty()) {
    throw std::runtime_error("cannot write the measurements into " +
                             std::string(output_directory));
  }
}

void profile_to_report(mlx_perf_build_graph_fn build_graph, void* builder_payload,
                       const char* variant_name, const char* report_title,
                       const char* output_directory, const char* report_file_base,
                       mlx_perf_job_summary* summary) {
  require_arguments(build_graph,
                    {variant_name, report_title, output_directory, report_file_base},
                    "mlx_perf_profile_to_report");
  const VariantMeasurement variant_measurement =
      measure_callback_job(build_graph, builder_payload, variant_name);
  const ReportInputs report_inputs{
      report_title, variant_measurement.timeline_graph,
      variant_measurement.gpu_timeline, variant_measurement.serial_telemetry,
      variant_measurement.job_totals};
  const ReportPaths report_paths =
      write_report(report_inputs, output_directory, report_file_base);
  if (!report_paths.is_written) {
    throw std::runtime_error("cannot write the report into " +
                             std::string(output_directory));
  }
  write_measurements_or_throw(variant_measurement, output_directory, report_file_base);
  if (summary != nullptr) {
    fill_summary(variant_measurement, *summary);
  }
}

void profile_to_measurements(mlx_perf_build_graph_fn build_graph, void* builder_payload,
                             const char* variant_name, const char* output_directory,
                             const char* measurements_file_base,
                             mlx_perf_job_summary* summary) {
  require_arguments(build_graph, {variant_name, output_directory, measurements_file_base},
                    "mlx_perf_profile_to_measurements");
  const VariantMeasurement variant_measurement =
      measure_callback_job(build_graph, builder_payload, variant_name);
  write_measurements_or_throw(variant_measurement, output_directory, measurements_file_base);
  if (summary != nullptr) {
    fill_summary(variant_measurement, *summary);
  }
}

// Variant names become file names (<base>_<name>.html), so an empty name, a
// '/', or two equal names would write files that collide or go astray.
void require_comparable_variants(const mlx_perf_variant* baseline,
                                 const mlx_perf_variant* candidate) {
  if (baseline == nullptr || candidate == nullptr) {
    throw std::invalid_argument("mlx_perf_compare_to_report: null variant");
  }
  for (const mlx_perf_variant* compared_variant : {baseline, candidate}) {
    require_arguments(compared_variant->build_graph, {compared_variant->variant_name},
                      "mlx_perf_compare_to_report");
    const std::string variant_name = compared_variant->variant_name;
    if (variant_name.empty() || variant_name.find('/') != std::string::npos) {
      throw std::invalid_argument("mlx_perf_compare_to_report: variant name \"" +
                                  variant_name + "\" must be non-empty without '/'");
    }
  }
  if (std::string(baseline->variant_name) == candidate->variant_name) {
    throw std::invalid_argument("mlx_perf_compare_to_report: both variants are named \"" +
                                std::string(baseline->variant_name) + "\"");
  }
}

// Each variant's streams are discovered on its own throwaway graph (timed
// like every other build); the pair is measured on their union.
void discover_streams(TimedGraphFactory& discovery_factory,
                      std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream) {
  discovery_factory.begin_pass("stream_discovery");
  for (const auto& [stream_index, stream] :
       collect_streams(capture_graph(discovery_factory.build_fresh_graph()))) {
    stream_index_to_stream.emplace(stream_index, stream);
  }
}

void prepend_build_passes(const TimedGraphFactory& discovery_factory,
                          VariantMeasurement& variant_measurement) {
  variant_measurement.graph_build_passes.insert(
      variant_measurement.graph_build_passes.begin(),
      discovery_factory.build_passes().begin(), discovery_factory.build_passes().end());
}

int to_c_verdict(ComparisonVerdict verdict) {
  switch (verdict) {
    case ComparisonVerdict::candidate_faster: return MLX_PERF_VERDICT_CANDIDATE_FASTER;
    case ComparisonVerdict::candidate_slower: return MLX_PERF_VERDICT_CANDIDATE_SLOWER;
    case ComparisonVerdict::within_spread: break;
  }
  return MLX_PERF_VERDICT_WITHIN_SPREAD;
}

void fill_metric(const PairedComparison& paired_comparison,
                 mlx_perf_metric_comparison& metric_comparison) {
  metric_comparison.baseline_median_us = paired_comparison.baseline.median;
  metric_comparison.baseline_min_us = paired_comparison.baseline.minimum;
  metric_comparison.baseline_max_us = paired_comparison.baseline.maximum;
  metric_comparison.candidate_median_us = paired_comparison.candidate.median;
  metric_comparison.candidate_min_us = paired_comparison.candidate.minimum;
  metric_comparison.candidate_max_us = paired_comparison.candidate.maximum;
  metric_comparison.median_difference_us = paired_comparison.median_difference;
  metric_comparison.relative_difference = paired_comparison.relative_difference;
  metric_comparison.verdict = to_c_verdict(paired_comparison.verdict);
  metric_comparison.rounds_candidate_faster = paired_comparison.rounds_candidate_faster;
  metric_comparison.round_count = paired_comparison.round_count;
}

void fill_comparison_summary(const PairedMeasurement& paired_measurement,
                             mlx_perf_comparison_summary& comparison_summary) {
  fill_summary(paired_measurement.baseline, comparison_summary.baseline);
  fill_summary(paired_measurement.candidate, comparison_summary.candidate);
  const PairComparisons pair_comparisons = compare_pair(paired_measurement);
  fill_metric(pair_comparisons.job_wall, comparison_summary.job_wall);
  fill_metric(pair_comparisons.gpu_execution, comparison_summary.gpu_execution);
  fill_metric(pair_comparisons.host_submission, comparison_summary.host_submission);
  comparison_summary.max_absolute_output_difference =
      paired_measurement.output_agreement.max_absolute_difference;
}

void compare_to_report(const mlx_perf_variant* baseline, const mlx_perf_variant* candidate,
                       const char* report_title, const char* output_directory,
                       const char* report_file_base, mlx_perf_comparison_summary* summary) {
  require_comparable_variants(baseline, candidate);
  require_arguments(baseline->build_graph, {report_title, output_directory, report_file_base},
                    "mlx_perf_compare_to_report");
  const CallbackGraphFactory baseline_factory(baseline->build_graph, baseline->builder_payload);
  const CallbackGraphFactory candidate_factory(candidate->build_graph,
                                               candidate->builder_payload);
  TimedGraphFactory baseline_discovery(baseline_factory);
  TimedGraphFactory candidate_discovery(candidate_factory);
  std::unordered_map<int, mlx::core::Stream> stream_index_to_stream;
  discover_streams(baseline_discovery, stream_index_to_stream);
  discover_streams(candidate_discovery, stream_index_to_stream);
  PairedMeasurement paired_measurement =
      measure_pair({baseline->variant_name, baseline_factory},
                   {candidate->variant_name, candidate_factory}, stream_index_to_stream);
  prepend_build_passes(baseline_discovery, paired_measurement.baseline);
  prepend_build_passes(candidate_discovery, paired_measurement.candidate);
  if (!write_report_bundle({report_title, output_directory, report_file_base},
                           paired_measurement)) {
    throw std::runtime_error("cannot write the comparison into " +
                             std::string(output_directory));
  }
  if (summary != nullptr) {
    fill_comparison_summary(paired_measurement, *summary);
  }
}

// Called only from inside a catch block: records the exception in flight
// as the thread's last error and returns the C failure status. Exceptions
// must not cross into C or Rust frames (undefined behavior), so every entry
// point catches everything and reports through here.
int record_failure_in_flight() {
  try {
    throw;
  } catch (const std::exception& failure) {
    last_error_message = failure.what();
  } catch (...) {
    last_error_message = "unknown failure";
  }
  return 1;
}

} // namespace

extern "C" int mlx_perf_profile_to_report(
    mlx_perf_build_graph_fn build_graph, void* builder_payload, const char* variant_name,
    const char* report_title, const char* output_directory, const char* report_file_base,
    mlx_perf_job_summary* summary) {
  last_error_message.clear();
  try {
    profile_to_report(build_graph, builder_payload, variant_name, report_title,
                      output_directory, report_file_base, summary);
    return 0;
  } catch (...) {
    return record_failure_in_flight();
  }
}

extern "C" int mlx_perf_profile_to_measurements(
    mlx_perf_build_graph_fn build_graph, void* builder_payload, const char* variant_name,
    const char* output_directory, const char* measurements_file_base,
    mlx_perf_job_summary* summary) {
  last_error_message.clear();
  try {
    profile_to_measurements(build_graph, builder_payload, variant_name, output_directory,
                            measurements_file_base, summary);
    return 0;
  } catch (...) {
    return record_failure_in_flight();
  }
}

extern "C" int mlx_perf_compare_to_report(
    const mlx_perf_variant* baseline, const mlx_perf_variant* candidate,
    const char* report_title, const char* output_directory, const char* report_file_base,
    mlx_perf_comparison_summary* summary) {
  last_error_message.clear();
  try {
    compare_to_report(baseline, candidate, report_title, output_directory, report_file_base,
                      summary);
    return 0;
  } catch (...) {
    return record_failure_in_flight();
  }
}

extern "C" const char* mlx_perf_version(void) { return MLX_PERF_VERSION_STRING; }

extern "C" const char* mlx_perf_last_error(void) { return last_error_message.c_str(); }
