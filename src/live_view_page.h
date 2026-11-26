#pragma once

// Live view page: one concern only — the single HTML shell the server
// returns at "/": Spark-UI-like tabs (Job / Stages / Tasks / Memory), a
// status line, a re-run button, and the small script that loads tab
// fragments and follows the server's event stream.

#include <string>

namespace mlx_perf {

std::string build_live_view_page(const std::string& job_title);

} // namespace mlx_perf
