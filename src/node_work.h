#pragma once

// Node work: one concern only — what one operation has to move and compute,
// as plain numbers. Filled by node_work_estimate from the captured graph and
// judged against the machine's ceilings by work_limits; free of MLX so both
// the judgement and the reports are tested without a GPU.

#include <cstdint>

namespace mlx_perf {

struct NodeWork {
  // Every distinct input at its stored size, and every output. Counted as
  // traffic to and from unified memory: a small intermediate that stays in a
  // cache between kernels is counted all the same, so for small tensors the
  // figure can sit above the true traffic. Weights dominate the large
  // kernels, where it is close.
  std::uint64_t bytes_read = 0;
  std::uint64_t bytes_written = 0;
  // Two operations per multiply-add. Counted for the matrix-multiply family
  // and fused attention only; other operations leave it uncounted rather
  // than guess, and a reader sees null.
  bool is_arithmetic_counted = false;
  double arithmetic_operations = 0.0;
  // A gathered multiply (GatherQMM, GatherMM) reads only the experts its
  // indices name. Until those indices are read (after evaluation),
  // bytes_read counts every expert and gathered_experts_used stays 0.
  bool is_gathered = false;
  int gathered_experts_total = 0;
  int gathered_experts_used = 0;
  std::uint64_t gathered_weight_bytes_all_experts = 0;
  std::uintptr_t gathered_index_array_id = 0;
};

// Adds `addend` into `total`: a stage's work is its members' work. The
// arithmetic total holds the members that counted theirs and is counted when
// any did. Members left uncounted (elementwise, normalization) do little
// arithmetic beside a stage's multiplies, so the total stays a fair floor.
void add_node_work(NodeWork& total, const NodeWork& addend);

} // namespace mlx_perf
