#pragma once

// JSON writer: one concern only — serialise a JsonValue tree to text. The
// parser (json_value) reads model configs; this writes the measurement file
// callers in other languages read, so its output must parse anywhere.

#include "json_value.h"

#include <string>

namespace mlx_perf {

// Two-space indentation, one member per line, members in key order (the
// tree's Object is an ordered map, so output is deterministic). Numbers use
// the shortest text that reads back to the same double; NaN and infinities,
// which JSON cannot represent, are written as null. Strings are UTF-8 with
// quotes, backslashes and control characters escaped.
std::string write_json(const JsonValue& document);

} // namespace mlx_perf
