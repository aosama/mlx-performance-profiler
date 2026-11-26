#include "measurement_document.h"

#include "mlx_perf/mlx_perf_version.h"
#include "timed_units.h"
#include "timeline_dependencies.h"

#include "mlx/version.h"

#include <map>
#include <string>
#include <vector>

namespace mlx_perf {
namespace {

using Object = JsonValue::Object;
using Array = JsonValue::Array;

JsonValue json_number(double number_value) { return JsonValue(number_value); }
JsonValue json_text(const std::string& text_value) { return JsonValue(text_value); }

JsonValue number_or_null(bool is_measured, double number_value) {
  return is_measured ? JsonValue(number_value) : JsonValue();
}

JsonValue position_or_null(int position) {
  return position == NO_POSITION ? JsonValue() : json_number(position);
}

template <typename Number>
JsonValue number_array(const std::vector<Number>& numbers) {
  Array number_values;
  for (const Number number_value : numbers) {
    number_values.push_back(json_number(static_cast<double>(number_value)));
  }
  return JsonValue(std::move(number_values));
}

// The document names slices the way the reports do ("wait on host"), not by
// the internal enum, so a reader needs no glossary of profiler internals.
std::string classification_name(NodeClass node_class) {
  switch (node_class) {
    case NodeClass::wait_on_cpu: return "wait_on_host";
    case NodeClass::wait_on_gpu: return "wait_on_own_stream";
    case NodeClass::wait_on_dependency: return "wait_on_other_stream";
    case NodeClass::scheduler_delay: return "gpu_scheduling";
    case NodeClass::compute: break;
  }
  return "compute";
}

// The timing every timed unit (a node, or a stage task) carries. Times are
// microseconds since the timeline pass started; the six slices sum to wall.
void add_unit_timing(const NodeTimestamps& timestamps, const LatencyBreakdown& breakdown,
                     Object& unit_record) {
  unit_record["issue_started_us"] = json_number(timestamps.issue_started_microseconds);
  unit_record["issue_finished_us"] = json_number(timestamps.issue_finished_microseconds);
  unit_record["completed_us"] = json_number(timestamps.completed_microseconds);
  unit_record["gpu_started_us"] =
      number_or_null(timestamps.has_gpu_window, timestamps.gpu_started_microseconds);
  unit_record["gpu_finished_us"] =
      number_or_null(timestamps.has_gpu_window, timestamps.gpu_finished_microseconds);
  unit_record["wait_on_host_us"] = json_number(breakdown.wait_on_cpu_microseconds);
  unit_record["wait_on_own_stream_us"] = json_number(breakdown.wait_on_gpu_microseconds);
  unit_record["wait_on_other_stream_us"] = json_number(breakdown.wait_on_dependency_microseconds);
  unit_record["gpu_scheduling_us"] = json_number(breakdown.scheduler_delay_microseconds);
  unit_record["compute_us"] = json_number(breakdown.compute_microseconds);
  unit_record["result_latency_us"] = json_number(breakdown.completion_latency_microseconds);
  unit_record["wall_us"] = json_number(breakdown.wall_microseconds);
  unit_record["is_compute_gpu_measured"] = JsonValue(breakdown.is_compute_measured);
  unit_record["classification"] = json_text(classification_name(breakdown.classification));
}

// A unit's GPU start minus the GPU end of the unit before it on the same
// stream: how long that stream's GPU queue sat idle in between. Null for the
// first unit on a stream, or when either side has no GPU window.
std::vector<JsonValue> compute_gpu_idle_before(const std::vector<TimedUnit>& timed_units) {
  std::vector<JsonValue> unit_to_gpu_idle_before(timed_units.size());
  std::map<int, const NodeTimestamps*> stream_index_to_previous_unit;
  for (std::size_t unit_index = 0; unit_index < timed_units.size(); ++unit_index) {
    const TimedUnit& timed_unit = timed_units[unit_index];
    const auto previous_entry = stream_index_to_previous_unit.find(timed_unit.stream_index);
    if (previous_entry != stream_index_to_previous_unit.end() &&
        previous_entry->second->has_gpu_window && timed_unit.timestamps->has_gpu_window) {
      unit_to_gpu_idle_before[unit_index] =
          json_number(timed_unit.timestamps->gpu_started_microseconds -
                 previous_entry->second->gpu_finished_microseconds);
    }
    stream_index_to_previous_unit[timed_unit.stream_index] = timed_unit.timestamps;
  }
  return unit_to_gpu_idle_before;
}

// Identity and structure of one node, whichever granularity was timed.
Object build_node_identity(const NodeTimeline& node_timeline, std::size_t position) {
  const GraphNode& graph_node = *node_timeline.node;
  Object node_record;
  node_record["position"] = json_number(static_cast<double>(position));
  node_record["label"] = json_text(graph_node.label);
  node_record["op_name"] = json_text(graph_node.op_name);
  node_record["kernel_name"] =
      graph_node.kernel_name.empty() ? JsonValue() : json_text(graph_node.kernel_name);
  node_record["stream_index"] = json_number(graph_node.stream_index);
  node_record["shape"] = number_array(graph_node.shape_dimensions);
  node_record["dtype"] = json_text(graph_node.dtype_string);
  node_record["output_bytes"] = json_number(static_cast<double>(graph_node.output_bytes));
  node_record["input_positions"] = number_array(node_timeline.input_positions);
  node_record["cross_stream_input_positions"] =
      number_array(node_timeline.cross_stream_producer_positions);
  node_record["previous_on_stream_position"] =
      position_or_null(node_timeline.previous_on_stream_position);
  node_record["stage_position"] = position_or_null(node_timeline.stage_position);
  return node_record;
}

// The serial pass ran an identical graph; a position whose op does not match
// gets nulls rather than a number borrowed from another node.
void add_serial_measurement(const RunTelemetry& serial_telemetry, std::size_t position,
                            const std::string& op_name, Object& node_record) {
  const NodeTelemetry* serial_node = find_attributed_node(serial_telemetry, position, op_name);
  node_record["serial_duration_us"] =
      number_or_null(serial_node != nullptr,
                     serial_node != nullptr ? serial_node->duration_microseconds : 0.0);
  node_record["memory_added_bytes"] = number_or_null(
      serial_node != nullptr,
      serial_node != nullptr ? static_cast<double>(serial_node->memory_added_bytes) : 0.0);
}

// Per-stage passes time stage tasks, not nodes: a member's timing fields are
// null so a reader can never mistake the stage's numbers for the node's. The
// field names come from add_unit_timing itself, so a timing field added there
// is nulled here too and both granularities always carry the same keys.
void add_untimed_node_fields(Object& node_record) {
  Object timing_fields;
  add_unit_timing(NodeTimestamps{}, LatencyBreakdown{}, timing_fields);
  for (const auto& timing_field : timing_fields) {
    node_record[timing_field.first] = JsonValue();
  }
  node_record["gpu_idle_before_us"] = JsonValue();
}

JsonValue build_node_records(const VariantMeasurement& variant_measurement,
                             const std::vector<JsonValue>& unit_to_gpu_idle_before) {
  const GpuTimeline& gpu_timeline = variant_measurement.gpu_timeline;
  const bool is_node_timed = gpu_timeline.granularity == TimelineGranularity::per_node;
  Array node_records;
  for (std::size_t position = 0; position < gpu_timeline.nodes.size(); ++position) {
    const NodeTimeline& node_timeline = gpu_timeline.nodes[position];
    Object node_record = build_node_identity(node_timeline, position);
    if (is_node_timed) {
      add_unit_timing(node_timeline.timestamps, node_timeline.breakdown, node_record);
      node_record["gpu_idle_before_us"] = unit_to_gpu_idle_before[position];
    } else {
      add_untimed_node_fields(node_record);
    }
    add_serial_measurement(variant_measurement.serial_telemetry, position,
                           node_timeline.node->op_name, node_record);
    node_records.push_back(JsonValue(std::move(node_record)));
  }
  return JsonValue(std::move(node_records));
}

JsonValue build_stage_records(const GpuTimeline& gpu_timeline,
                              const std::vector<JsonValue>& unit_to_gpu_idle_before) {
  Array stage_records;
  for (std::size_t stage_position = 0; stage_position < gpu_timeline.stages.size();
       ++stage_position) {
    const StageTimeline& stage_task = gpu_timeline.stages[stage_position];
    Object stage_record;
    stage_record["stage_position"] = json_number(static_cast<double>(stage_position));
    stage_record["stage_index"] = json_number(stage_task.stage_index);
    stage_record["stage_name"] = json_text(stage_task.stage_name);
    stage_record["stream_index"] = json_number(stage_task.stream_index);
    stage_record["node_positions"] = number_array(stage_task.node_positions);
    stage_record["cross_stream_input_stage_positions"] =
        number_array(stage_task.cross_stream_producer_stage_positions);
    add_unit_timing(stage_task.timestamps, stage_task.breakdown, stage_record);
    stage_record["gpu_idle_before_us"] = unit_to_gpu_idle_before[stage_position];
    stage_records.push_back(JsonValue(std::move(stage_record)));
  }
  return JsonValue(std::move(stage_records));
}

JsonValue build_stream_records(const GpuTimeline& gpu_timeline) {
  Array stream_records;
  for (const StreamTimeline& stream_timeline : gpu_timeline.streams) {
    stream_records.push_back(JsonValue(Object{
        {"stream_index", json_number(stream_timeline.stream_index)},
        {"is_gpu", JsonValue(stream_timeline.is_gpu)},
        {"timed_unit_count", json_number(stream_timeline.node_count)},
        {"first_issue_us", json_number(stream_timeline.first_issue_microseconds)},
        {"last_completion_us", json_number(stream_timeline.last_completion_microseconds)},
        {"wall_us", json_number(stream_timeline.wall_microseconds)},
        {"busy_us", json_number(stream_timeline.busy_microseconds)},
        {"wait_on_other_stream_us", json_number(stream_timeline.dependency_wait_microseconds)}}));
  }
  return JsonValue(std::move(stream_records));
}

JsonValue build_graph_build_records(const std::vector<GraphBuildPass>& graph_build_passes) {
  Array pass_records;
  for (const GraphBuildPass& build_pass : graph_build_passes) {
    double total_microseconds = 0.0;
    for (const double call_microseconds : build_pass.call_microseconds) {
      total_microseconds += call_microseconds;
    }
    pass_records.push_back(JsonValue(Object{
        {"pass", json_text(build_pass.pass_name)},
        {"calls_us", number_array(build_pass.call_microseconds)},
        {"total_us", json_number(total_microseconds)}}));
  }
  return JsonValue(std::move(pass_records));
}

JsonValue build_job_record(const VariantMeasurement& variant_measurement,
                           std::size_t timed_unit_count) {
  const JobTotals& job_totals = variant_measurement.job_totals;
  const GpuTimeline& gpu_timeline = variant_measurement.gpu_timeline;
  return JsonValue(Object{
      {"job_wall_us", JsonValue(Object{
          {"median", json_number(job_totals.job_wall_microseconds)},
          {"min", json_number(job_totals.job_wall_min_microseconds)},
          {"max", json_number(job_totals.job_wall_max_microseconds)},
          {"samples", number_array(job_totals.job_wall_sample_microseconds)}})},
      {"timeline_wall_us", json_number(gpu_timeline.timeline_wall_microseconds)},
      {"sum_of_stages_us", json_number(job_totals.sum_of_stages_microseconds)},
      {"measured_gpu_execution_us", json_number(gpu_timeline.measured_gpu_compute_microseconds)},
      {"cross_stream_stall_us", json_number(gpu_timeline.cross_stream_stall_microseconds)},
      {"executed_node_count", json_number(static_cast<double>(gpu_timeline.nodes.size()))},
      {"gpu_measured_node_count", json_number(gpu_timeline.gpu_measured_node_count)},
      {"timed_unit_count", json_number(static_cast<double>(timed_unit_count))},
      {"stream_count", json_number(static_cast<double>(gpu_timeline.streams.size()))},
      {"peak_memory_bytes",
       json_number(static_cast<double>(variant_measurement.serial_telemetry.run_peak_memory_bytes))},
      {"command_buffers", JsonValue(Object{
          {"timeline_pass_observed", json_number(gpu_timeline.observed_command_buffer_count)},
          {"is_lower_bound", JsonValue(gpu_timeline.is_command_buffer_count_lower_bound)}})},
      {"graph_build", build_graph_build_records(variant_measurement.graph_build_passes)}});
}

} // namespace

JsonValue build_measurement_document(const VariantMeasurement& variant_measurement) {
  const GpuTimeline& gpu_timeline = variant_measurement.gpu_timeline;
  const bool is_stage_timed = gpu_timeline.granularity == TimelineGranularity::per_stage;
  const std::vector<TimedUnit> timed_units = collect_timed_units(gpu_timeline);
  const std::vector<JsonValue> unit_to_gpu_idle_before = compute_gpu_idle_before(timed_units);
  const std::vector<JsonValue> no_unit_timing;
  return JsonValue(Object{
      {"schema_version", json_number(MEASUREMENT_SCHEMA_VERSION)},
      {"mlx_perf_version", json_text(MLX_PERF_VERSION_STRING)},
      {"mlx_version", json_text(mlx::core::version())},
      // The only backend the probe measures today; Linux and CUDA are #15.
      {"backend", json_text("metal")},
      {"variant_name", json_text(variant_measurement.variant_name)},
      {"granularity", json_text(is_stage_timed ? "per_stage" : "per_node")},
      {"time_origin", json_text("microseconds since the timeline pass started")},
      {"job", build_job_record(variant_measurement, timed_units.size())},
      {"streams", build_stream_records(gpu_timeline)},
      {"nodes", build_node_records(variant_measurement,
                                   is_stage_timed ? no_unit_timing : unit_to_gpu_idle_before)},
      {"stages", build_stage_records(gpu_timeline,
                                     is_stage_timed ? unit_to_gpu_idle_before : no_unit_timing)}});
}

} // namespace mlx_perf
