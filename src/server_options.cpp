#include "server_options.h"

#include <cstdlib>

namespace mlx_perf {
namespace {

constexpr long MAX_PORT_NUMBER = 65535;

bool parse_port(const std::string& port_text, std::uint16_t& port) {
  if (port_text.empty()) {
    return false;
  }
  char* parse_end = nullptr;
  const long parsed_port = std::strtol(port_text.c_str(), &parse_end, 10);
  if (*parse_end != '\0' || parsed_port < 1 || parsed_port > MAX_PORT_NUMBER) {
    return false;
  }
  port = static_cast<std::uint16_t>(parsed_port);
  return true;
}

} // namespace

ServerOptions parse_server_options(const std::vector<std::string>& arguments) {
  ServerOptions server_options;
  for (std::size_t argument_index = 0; argument_index < arguments.size();
       ++argument_index) {
    const std::string& argument = arguments[argument_index];
    if (argument == "--help" || argument == "-h") {
      server_options.should_show_help = true;
    } else if (argument == "--port") {
      if (argument_index + 1 >= arguments.size() ||
          !parse_port(arguments[argument_index + 1], server_options.port)) {
        server_options.error_message = "--port needs a number from 1 to 65535";
        return server_options;
      }
      ++argument_index;
    } else {
      server_options.error_message = "unknown argument: " + argument;
      return server_options;
    }
  }
  return server_options;
}

std::string server_usage_text() {
  return "usage: mlx_perf_server [--port N]\n"
         "  Measures the demo job (eager and compiled) and serves a live,\n"
         "  Spark-UI-style view at http://127.0.0.1:N/ (default N = 8765).\n"
         "  Loopback only. Ctrl-C to stop.\n";
}

} // namespace mlx_perf
