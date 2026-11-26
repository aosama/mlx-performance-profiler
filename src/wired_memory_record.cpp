#include "wired_memory_record.h"

#include "wired_memory_analysis.h"

#include <optional>
#include <string>

namespace mlx_perf {
namespace {

using Object = JsonValue::Object;
using Array = JsonValue::Array;

JsonValue bytes(std::uint64_t byte_count) { return json_number(static_cast<double>(byte_count)); }

JsonValue signed_bytes(std::int64_t byte_count) { return json_number(static_cast<double>(byte_count)); }

JsonValue optional_number(const std::optional<double>& number_value) {
  return number_value.has_value() ? json_number(*number_value) : JsonValue();
}

JsonValue sample_record(const WiredMemorySample& wired_memory_sample, double first_sample_seconds) {
  const bool is_limit_known = wired_memory_sample.wire_limit_bytes > 0;
  return JsonValue(Object{
      {"time_us", json_number((wired_memory_sample.host_seconds - first_sample_seconds) * 1e6)},
      {"sample_point", json_text(std::string(wired_sample_point_name(wired_memory_sample.sample_point)))},
      {"system_wired_bytes", bytes(wired_memory_sample.system_wired_bytes)},
      {"wire_limit_bytes", is_limit_known ? bytes(wired_memory_sample.wire_limit_bytes) : JsonValue()},
      {"mlx_active_bytes", bytes(wired_memory_sample.mlx_active_bytes)},
      {"mlx_cache_bytes", bytes(wired_memory_sample.mlx_cache_bytes)},
      {"wired_outside_mlx_bytes", signed_bytes(wired_outside_mlx_bytes(wired_memory_sample))},
      {"wire_headroom_bytes", is_limit_known ? signed_bytes(wire_headroom_bytes(wired_memory_sample)) : JsonValue()}});
}

JsonValue release_event_record(const WireReleaseEvent& release_event) {
  return JsonValue(Object{
      {"released_sample_index", json_number(release_event.released_sample_index)},
      {"released_bytes", bytes(release_event.released_bytes)},
      {"unwired_sample_index", release_event.unwired_sample_index.has_value()
                                   ? json_number(*release_event.unwired_sample_index)
                                   : JsonValue()},
      {"release_lag_us", optional_number(release_event.release_lag_microseconds)}});
}

JsonValue index_array(const std::vector<int>& sample_indices) {
  Array index_values;
  for (const int sample_index : sample_indices) {
    index_values.push_back(json_number(sample_index));
  }
  return JsonValue(std::move(index_values));
}

JsonValue release_records(const std::vector<WireReleaseEvent>& release_events) {
  Array release_event_records;
  for (const WireReleaseEvent& release_event : release_events) {
    release_event_records.push_back(release_event_record(release_event));
  }
  return JsonValue(std::move(release_event_records));
}

} // namespace

JsonValue build_wired_memory_record(const std::vector<WiredMemorySample>& wired_memory_samples) {
  if (wired_memory_samples.empty()) {
    return JsonValue();
  }
  const WiredMemoryAnalysis wired_memory_analysis = analyze_wired_memory(wired_memory_samples);
  const bool is_limit_known = wired_memory_analysis.wire_limit_bytes > 0;
  Array sample_records;
  for (const WiredMemorySample& wired_memory_sample : wired_memory_samples) {
    sample_records.push_back(sample_record(wired_memory_sample, wired_memory_samples.front().host_seconds));
  }
  const std::optional<std::int64_t>& lowest_headroom = wired_memory_analysis.lowest_wire_headroom_bytes;
  return JsonValue(Object{
      {"wire_limit_bytes", is_limit_known ? bytes(wired_memory_analysis.wire_limit_bytes) : JsonValue()},
      {"sample_count", json_number(wired_memory_analysis.sample_count)},
      {"peak_system_wired_bytes", bytes(wired_memory_analysis.peak_system_wired_bytes)},
      {"peak_mlx_held_bytes", bytes(wired_memory_analysis.peak_mlx_held_bytes)},
      {"peak_wired_outside_mlx_bytes", signed_bytes(wired_memory_analysis.peak_wired_outside_mlx_bytes)},
      {"lowest_wired_outside_mlx_bytes", signed_bytes(wired_memory_analysis.lowest_wired_outside_mlx_bytes)},
      {"lowest_wire_headroom_bytes", lowest_headroom.has_value() ? signed_bytes(*lowest_headroom) : JsonValue()},
      {"has_crossed_wire_limit", is_limit_known ? JsonValue(wired_memory_analysis.has_crossed_wire_limit) : JsonValue()},
      {"release_events", release_records(wired_memory_analysis.release_events)},
      {"median_release_lag_us", optional_number(median_release_lag_microseconds(wired_memory_analysis.release_events))},
      {"largest_release_lag_us", optional_number(largest_release_lag_microseconds(wired_memory_analysis.release_events))},
      {"release_lag_inflated_sample_indices", index_array(wired_memory_analysis.release_lag_inflated_sample_indices)},
      {"largest_release_lag_inflation_bytes", bytes(wired_memory_analysis.largest_release_lag_inflation_bytes)},
      {"transient_jump_sample_indices", index_array(wired_memory_analysis.transient_jump_sample_indices)},
      {"samples", JsonValue(std::move(sample_records))}});
}

} // namespace mlx_perf
