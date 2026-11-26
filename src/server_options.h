#pragma once

// Server options: one concern only — parse mlx_perf_server's command line.
// Configuration is flags, never environment variables, so every run states
// its settings where they can be seen.

#include <cstdint>
#include <string>
#include <vector>

namespace mlx_perf {

struct ServerOptions {
  std::uint16_t port = 8765;
  bool should_show_help = false;
  std::string error_message;  // non-empty when the command line was invalid
};

ServerOptions parse_server_options(const std::vector<std::string>& arguments);
std::string server_usage_text();

} // namespace mlx_perf
