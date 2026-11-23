// Integration tests for http_server over real loopback sockets (no GPU).

#include "http_server.h"
#include "http_test_client.h"
#include "test_harness.h"

#include <string>

namespace {

using namespace mlx_plan;

class EchoHandler final : public HttpHandler {
 public:
  void handle(const HttpRequest& request, HttpConnection& connection) override {
    connection.send_response(200, "text/plain",
                             request.method + " " + request.path + " ?" + request.query);
  }
};

bool starts_with(const std::string& text, const std::string& prefix) {
  return text.rfind(prefix, 0) == 0;
}

// --- positive ---

void should_parse_method_path_and_query() {
  HttpRequest request;
  EXPECT_TRUE(parse_request_line("GET /view/tasks?variant=eager HTTP/1.1", request));
  EXPECT_TRUE(request.method == "GET");
  EXPECT_TRUE(request.path == "/view/tasks");
  EXPECT_TRUE(request.query == "variant=eager");
}

void should_serve_a_request_on_an_ephemeral_loopback_port() {
  EchoHandler echo_handler;
  HttpServer http_server(echo_handler);
  const std::uint16_t bound_port = http_server.start(0);
  EXPECT_TRUE(bound_port != 0);
  const std::string response_text = http_test_client::get(bound_port, "/hello?x=1");
  EXPECT_TRUE(starts_with(response_text, "HTTP/1.1 200 OK"));
  EXPECT_TRUE(response_text.find("GET /hello ?x=1") != std::string::npos);
  EXPECT_TRUE(response_text.find("Content-Length: 15") != std::string::npos);
  http_server.stop();
}

// --- edge / boundary ---

void should_parse_a_path_without_query() {
  HttpRequest request;
  EXPECT_TRUE(parse_request_line("POST / HTTP/1.0", request));
  EXPECT_TRUE(request.path == "/" && request.query.empty());
}

void should_stop_idempotently_and_refuse_new_connections() {
  EchoHandler echo_handler;
  HttpServer http_server(echo_handler);
  const std::uint16_t bound_port = http_server.start(0);
  http_server.stop();
  http_server.stop();
  EXPECT_TRUE(http_test_client::get(bound_port, "/").empty());
}

// --- negative ---

void should_reject_malformed_request_lines() {
  HttpRequest request;
  EXPECT_TRUE(!parse_request_line("", request));
  EXPECT_TRUE(!parse_request_line("GET", request));
  EXPECT_TRUE(!parse_request_line("GET relative HTTP/1.1", request));
  EXPECT_TRUE(!parse_request_line("GET / SPDY/3", request));
}

void should_answer_400_to_garbage_and_431_to_oversized_heads() {
  EchoHandler echo_handler;
  HttpServer http_server(echo_handler);
  const std::uint16_t bound_port = http_server.start(0);
  EXPECT_TRUE(starts_with(
      http_test_client::send_raw_request(bound_port, "NONSENSE\r\n\r\n"), "HTTP/1.1 400"));
  const std::string oversized_request =
      "GET / HTTP/1.1\r\nX-Padding: " + std::string(20000, 'a') + "\r\n\r\n";
  EXPECT_TRUE(starts_with(http_test_client::send_raw_request(bound_port, oversized_request),
                          "HTTP/1.1 431"));
  http_server.stop();
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_parse_method_path_and_query),
      TEST_CASE(should_serve_a_request_on_an_ephemeral_loopback_port),
      TEST_CASE(should_parse_a_path_without_query),
      TEST_CASE(should_stop_idempotently_and_refuse_new_connections),
      TEST_CASE(should_reject_malformed_request_lines),
      TEST_CASE(should_answer_400_to_garbage_and_431_to_oversized_heads),
  });
}
