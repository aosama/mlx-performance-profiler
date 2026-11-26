#pragma once

// Qwen3.5 generation: one concern only — greedy decoding with the real
// decode step, used to prove the forward math correct before any of its
// timings are trusted.

#include "qwen35_decode_step.h"

#include <vector>

namespace mlx_perf::qwen35 {

// Feeds `prompt_token_ids` one token at a time (decode steps also serve as
// prefill), then appends `new_token_count` argmax tokens. Returns only the
// generated ids.
std::vector<int> generate_greedy(const Qwen35Weights& weights, const Qwen35TextConfig& config,
                                 const std::vector<int>& prompt_token_ids, int new_token_count);

} // namespace mlx_perf::qwen35
