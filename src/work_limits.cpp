#include "work_limits.h"

#include <algorithm>

namespace mlx_perf {
namespace {

constexpr double MICROSECONDS_PER_SECOND = 1e6;

void price_work_at_ceilings(const NodeWork& unit_work, const MachineCeilings& machine_ceilings,
                            WorkJudgement& work_judgement) {
  const double bytes_moved = static_cast<double>(unit_work.bytes_read + unit_work.bytes_written);
  work_judgement.has_data_movement_time = true;
  work_judgement.data_movement_microseconds =
      bytes_moved / machine_ceilings.memory_read_bytes_per_second * MICROSECONDS_PER_SECOND;
  if (unit_work.is_arithmetic_counted) {
    work_judgement.has_arithmetic_time = true;
    work_judgement.arithmetic_microseconds = unit_work.arithmetic_operations /
        machine_ceilings.arithmetic_operations_per_second * MICROSECONDS_PER_SECOND;
  }
}

WorkLimit name_the_limit(const WorkJudgement& work_judgement) {
  const double larger_share =
      std::max(work_judgement.data_movement_share, work_judgement.arithmetic_share);
  if (larger_share < LIMITING_SHARE_THRESHOLD) {
    return WorkLimit::neither;
  }
  return work_judgement.arithmetic_share > work_judgement.data_movement_share
      ? WorkLimit::arithmetic : WorkLimit::data_movement;
}

} // namespace

WorkJudgement judge_work(const NodeWork& unit_work, const MachineCeilings& machine_ceilings,
                         double gpu_compute_microseconds, bool is_compute_gpu_measured) {
  WorkJudgement work_judgement;
  const bool has_work = unit_work.bytes_read + unit_work.bytes_written > 0;
  if (!machine_ceilings.is_measured || !has_work) {
    return work_judgement;
  }
  price_work_at_ceilings(unit_work, machine_ceilings, work_judgement);
  if (!is_compute_gpu_measured || gpu_compute_microseconds <= 0.0) {
    return work_judgement;
  }
  work_judgement.has_shares = true;
  work_judgement.data_movement_share =
      work_judgement.data_movement_microseconds / gpu_compute_microseconds;
  work_judgement.arithmetic_share = work_judgement.has_arithmetic_time
      ? work_judgement.arithmetic_microseconds / gpu_compute_microseconds : 0.0;
  work_judgement.limited_by = name_the_limit(work_judgement);
  return work_judgement;
}

std::string_view work_limit_name(WorkLimit work_limit) {
  switch (work_limit) {
    case WorkLimit::data_movement:
      return "data_movement";
    case WorkLimit::arithmetic:
      return "arithmetic";
    case WorkLimit::neither:
      return "neither";
    case WorkLimit::not_judged:
      return "";
  }
  return "";
}

} // namespace mlx_perf
