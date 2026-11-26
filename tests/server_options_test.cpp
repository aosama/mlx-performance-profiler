// Unit tests for server_options: the mlx_perf_server command line.

#include "server_options.h"
#include "test_harness.h"

namespace {

using mlx_perf::parse_server_options;

// --- positive ---

void should_default_to_port_8765() {
  const auto server_options = parse_server_options({});
  EXPECT_TRUE(server_options.port == 8765);
  EXPECT_TRUE(server_options.error_message.empty());
}

void should_accept_an_explicit_port() {
  EXPECT_TRUE(parse_server_options({"--port", "9000"}).port == 9000);
}

void should_recognize_help() {
  EXPECT_TRUE(parse_server_options({"--help"}).should_show_help);
  EXPECT_TRUE(parse_server_options({"-h"}).should_show_help);
}

// --- edge / boundary ---

void should_accept_the_lowest_and_highest_ports() {
  EXPECT_TRUE(parse_server_options({"--port", "1"}).port == 1);
  EXPECT_TRUE(parse_server_options({"--port", "65535"}).port == 65535);
}

// --- negative ---

void should_reject_out_of_range_or_non_numeric_ports() {
  for (const char* bad_port : {"0", "65536", "-1", "80x", ""}) {
    EXPECT_TRUE(!parse_server_options({"--port", bad_port}).error_message.empty());
  }
}

void should_reject_a_missing_port_value_and_unknown_flags() {
  EXPECT_TRUE(!parse_server_options({"--port"}).error_message.empty());
  EXPECT_TRUE(!parse_server_options({"--verbose"}).error_message.empty());
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_default_to_port_8765),
      TEST_CASE(should_accept_an_explicit_port),
      TEST_CASE(should_recognize_help),
      TEST_CASE(should_accept_the_lowest_and_highest_ports),
      TEST_CASE(should_reject_out_of_range_or_non_numeric_ports),
      TEST_CASE(should_reject_a_missing_port_value_and_unknown_flags),
  });
}
