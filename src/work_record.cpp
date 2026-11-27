#include "work_record.h"

#include <string>

namespace mlx_perf {
namespace {

using Object = JsonValue::Object;

JsonValue number_or_null(bool is_known, double number_value) {
  return is_known ? json_number(number_value) : JsonValue();
}

void add_gathered_experts(const NodeWork& unit_work, Object& work_record) {
  work_record["gathered_experts_total"] = number_or_null(
      unit_work.is_gathered, static_cast<double>(unit_work.gathered_experts_total));
  work_record["gathered_experts_used"] = number_or_null(
      unit_work.gathered_experts_used > 0, static_cast<double>(unit_work.gathered_experts_used));
}

void add_judgement(const WorkJudgement& work_judgement, Object& work_record) {
  work_record["data_movement_at_ceiling_us"] = number_or_null(
      work_judgement.has_data_movement_time, work_judgement.data_movement_microseconds);
  work_record["arithmetic_at_ceiling_us"] = number_or_null(
      work_judgement.has_arithmetic_time, work_judgement.arithmetic_microseconds);
  work_record["data_movement_share"] =
      number_or_null(work_judgement.has_shares, work_judgement.data_movement_share);
  work_record["arithmetic_share"] = number_or_null(
      work_judgement.has_shares && work_judgement.has_arithmetic_time, work_judgement.arithmetic_share);
  const std::string limit_name(work_limit_name(work_judgement.limited_by));
  work_record["limited_by"] = limit_name.empty() ? JsonValue() : json_text(limit_name);
}

} // namespace

JsonValue build_machine_ceilings_record(const MachineCeilings& machine_ceilings) {
  if (!machine_ceilings.is_measured) {
    return JsonValue();
  }
  return JsonValue(Object{
      {"memory_read_bytes_per_second", json_number(machine_ceilings.memory_read_bytes_per_second)},
      {"arithmetic_operations_per_second", json_number(machine_ceilings.arithmetic_operations_per_second)},
      {"arithmetic_dtype", json_text(machine_ceilings.arithmetic_dtype)}});
}

JsonValue build_work_record(const NodeWork& unit_work, const WorkJudgement& work_judgement) {
  if (unit_work.bytes_read + unit_work.bytes_written == 0) {
    return JsonValue();
  }
  Object work_record;
  work_record["bytes_read"] = json_number(static_cast<double>(unit_work.bytes_read));
  work_record["bytes_written"] = json_number(static_cast<double>(unit_work.bytes_written));
  work_record["arithmetic_operations"] =
      number_or_null(unit_work.is_arithmetic_counted, unit_work.arithmetic_operations);
  add_gathered_experts(unit_work, work_record);
  add_judgement(work_judgement, work_record);
  return JsonValue(std::move(work_record));
}

} // namespace mlx_perf
