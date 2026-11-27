#pragma once

// Work limits: one concern only — judge one timed unit (a node or a stage
// task) against the machine's ceilings: how long its bytes would take at the
// reading ceiling, how long its arithmetic would take at the arithmetic
// ceiling, each as a share of the GPU time it actually took, and which one
// held it back. Pure arithmetic; the reports format the judgement.
//
// Why two shares and not one "time spent moving data": inside a kernel the
// GPU moves data and computes at the same time, many threadgroups covering
// each other's waits, so no clock can split a kernel's time between the
// two. What can be said exactly is how much of the time each would need on
// its own at full speed; the larger share names the limit.

#include "machine_ceilings.h"
#include "node_work.h"

#include <string_view>

namespace mlx_perf {

enum class WorkLimit {
  data_movement,
  arithmetic,
  // Neither need reaches half the time taken: the GPU was not kept busy
  // (too few threadgroups, launch and synchronization costs, a serial
  // dependency inside the kernel). Often where a kernel can be made faster.
  neither,
  // No GPU window, no work, or no ceilings: nothing to judge.
  not_judged,
};

// A unit is limited by data movement or arithmetic when that need fills at
// least half its time. Below half, more than half the time went to
// something other than the limit, which is the more useful thing to say.
inline constexpr double LIMITING_SHARE_THRESHOLD = 0.5;

struct WorkJudgement {
  bool has_data_movement_time = false;
  double data_movement_microseconds = 0.0;
  bool has_arithmetic_time = false;
  double arithmetic_microseconds = 0.0;
  // Each need over the GPU-measured compute; may exceed 1 when a kernel
  // beats the calibration (see machine_ceilings.h).
  bool has_shares = false;
  double data_movement_share = 0.0;
  double arithmetic_share = 0.0;
  WorkLimit limited_by = WorkLimit::not_judged;
};

// Bytes written are priced at the reading ceiling too: writes go to the same
// memory over the same link, and the calibration measures one direction.
WorkJudgement judge_work(const NodeWork& unit_work, const MachineCeilings& machine_ceilings,
                         double gpu_compute_microseconds, bool is_compute_gpu_measured);

// "data_movement", "arithmetic", "neither"; "" for not_judged.
std::string_view work_limit_name(WorkLimit work_limit);

} // namespace mlx_perf
