#include "run_memory_record.h"

namespace mlx_perf {
namespace {

using Object = JsonValue::Object;
using Array = JsonValue::Array;

JsonValue json_byte_count(std::uint64_t byte_count) { return json_number(static_cast<double>(byte_count)); }

JsonValue json_signed_byte_count(std::int64_t byte_count) { return json_number(static_cast<double>(byte_count)); }

} // namespace

JsonValue build_run_memory_record(const RunMemoryFigures& memory_figures) {
  const bool is_limit_known = memory_figures.wire_limit_bytes > 0;
  const bool is_pressure_known = memory_figures.are_pressure_counters_readable;
  return JsonValue(Object{
      {"peak_mlx_active_bytes", json_byte_count(memory_figures.peak_mlx_active_bytes)},
      {"peak_mlx_active_added_bytes", json_byte_count(memory_figures.peak_mlx_active_added_bytes)},
      {"mlx_held_at_end_bytes", json_byte_count(memory_figures.mlx_held_at_end_bytes)},
      {"mlx_held_change_bytes", json_signed_byte_count(memory_figures.mlx_held_change_bytes)},
      {"system_wired_at_end_bytes", json_byte_count(memory_figures.system_wired_at_end_bytes)},
      {"wire_limit_bytes", is_limit_known ? json_byte_count(memory_figures.wire_limit_bytes) : JsonValue()},
      {"wire_requests_over_limit_count",
       is_pressure_known ? json_byte_count(memory_figures.wire_requests_over_limit_count) : JsonValue()},
      {"compressed_bytes", is_pressure_known ? json_byte_count(memory_figures.compressed_bytes) : JsonValue()},
      {"swap_used_change_bytes", is_pressure_known ? json_signed_byte_count(memory_figures.swap_used_change_bytes) : JsonValue()},
      {"has_crossed_wire_limit",
       is_pressure_known || is_limit_known ? JsonValue(has_crossed_wire_limit(memory_figures)) : JsonValue()},
      {"has_compressed_memory", is_pressure_known ? JsonValue(has_compressed_memory(memory_figures)) : JsonValue()}});
}

JsonValue build_rounds_memory_record(const std::vector<RunMemoryFigures>& round_memory_figures) {
  Array round_records;
  for (const RunMemoryFigures& memory_figures : round_memory_figures) {
    round_records.push_back(build_run_memory_record(memory_figures));
  }
  return JsonValue(Object{
      {"rounds_crossing_wire_limit", json_number(count_runs_crossing_wire_limit(round_memory_figures))},
      {"rounds_with_compression", json_number(count_runs_with_compression(round_memory_figures))},
      {"rounds", JsonValue(std::move(round_records))}});
}

} // namespace mlx_perf
