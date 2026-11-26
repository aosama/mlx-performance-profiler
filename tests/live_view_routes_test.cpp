// Component tests for live_view_routes: the routes served over a real
// loopback server, backed by a store pre-filled with fake views (no GPU).

#include "http_server.h"
#include "http_test_client.h"
#include "live_view_routes.h"
#include "live_view_store.h"
#include "test_harness.h"

#include <chrono>
#include <string>

namespace {

using namespace mlx_perf;

bool contains(const std::string& text, const std::string& needle) {
  return text.find(needle) != std::string::npos;
}

struct RunningLiveServer {
  LiveViewStore live_view_store;
  LiveViewRoutes live_view_routes{live_view_store, "Test <job>"};
  HttpServer http_server{live_view_routes};
  std::uint16_t bound_port = 0;

  RunningLiveServer() {
    live_view_store.publish_views({"<p>JOB</p>", "<p>STAGES</p>", "<p>TASKS</p>",
                                   "<p>MEMORY</p>"});
    live_view_store.set_status("idle \"quoted\"");
    bound_port = http_server.start(0);
  }
  ~RunningLiveServer() {
    live_view_routes.close_event_streams();
    http_server.stop();
  }
};

// --- positive ---

void should_serve_the_shell_with_all_four_tabs() {
  RunningLiveServer live_server;
  const std::string shell_page = http_test_client::get(live_server.bound_port, "/");
  for (const char* tab_name : {"job", "stages", "tasks", "memory"}) {
    EXPECT_TRUE(contains(shell_page, "data-tab=\"" + std::string(tab_name) + "\""));
  }
  EXPECT_TRUE(contains(shell_page, "Test &lt;job&gt;"));
}

void should_serve_each_tab_fragment_from_the_store() {
  RunningLiveServer live_server;
  EXPECT_TRUE(contains(http_test_client::get(live_server.bound_port, "/view/tasks"),
                       "<p>TASKS</p>"));
  const std::string memory_view =
      http_test_client::get(live_server.bound_port, "/view/memory");
  EXPECT_TRUE(contains(memory_view, "Process memory (live"));
  EXPECT_TRUE(contains(memory_view, "<p>MEMORY</p>"));
}

void should_report_status_as_valid_escaped_json() {
  RunningLiveServer live_server;
  const std::string status_response =
      http_test_client::get(live_server.bound_port, "/api/status");
  EXPECT_TRUE(contains(status_response, "application/json"));
  EXPECT_TRUE(contains(status_response, "\"status\":\"idle \\\"quoted\\\"\""));
  EXPECT_TRUE(contains(status_response, "\"views_version\":1"));
}

void should_accept_a_run_request_and_hand_it_to_the_main_thread() {
  RunningLiveServer live_server;
  EXPECT_TRUE(contains(http_test_client::post(live_server.bound_port, "/api/run"),
                       "HTTP/1.1 202"));
  EXPECT_TRUE(live_server.live_view_store.wait_for_run_request(
      std::chrono::milliseconds(100)));
}

// --- edge / boundary ---

void should_open_the_event_stream_with_an_initial_update() {
  RunningLiveServer live_server;
  // The stream never ends on its own; the client's receive timeout ends it.
  const std::string event_stream = http_test_client::send_raw_request(
      live_server.bound_port, "GET /events HTTP/1.1\r\n\r\n", 300);
  EXPECT_TRUE(contains(event_stream, "text/event-stream"));
  EXPECT_TRUE(contains(event_stream, "event: update\ndata: 2"));
}

// --- negative ---

void should_reject_unknown_paths_tabs_and_methods() {
  RunningLiveServer live_server;
  EXPECT_TRUE(contains(http_test_client::get(live_server.bound_port, "/nope"),
                       "HTTP/1.1 404"));
  EXPECT_TRUE(contains(http_test_client::get(live_server.bound_port, "/view/secrets"),
                       "HTTP/1.1 404"));
  EXPECT_TRUE(contains(http_test_client::get(live_server.bound_port, "/api/run"),
                       "HTTP/1.1 405"));
  EXPECT_TRUE(contains(http_test_client::post(live_server.bound_port, "/view/job"),
                       "HTTP/1.1 405"));
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_serve_the_shell_with_all_four_tabs),
      TEST_CASE(should_serve_each_tab_fragment_from_the_store),
      TEST_CASE(should_report_status_as_valid_escaped_json),
      TEST_CASE(should_accept_a_run_request_and_hand_it_to_the_main_thread),
      TEST_CASE(should_open_the_event_stream_with_an_initial_update),
      TEST_CASE(should_reject_unknown_paths_tabs_and_methods),
  });
}
