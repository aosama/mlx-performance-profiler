#pragma once

// HTTP server: one concern only — a minimal HTTP/1.1 listener on loopback
// that parses request lines and hands each connection to a handler. No
// framework: the live view needs four routes and one event stream, and a
// dependency would outweigh the ~200 lines it replaces.
//
// Security posture: binds 127.0.0.1 only (never all interfaces), reads at
// most MAX_REQUEST_HEAD_BYTES of request head, ignores request bodies.

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>

namespace mlx_perf {

struct HttpRequest {
  std::string method;  // "GET", "POST"
  std::string path;    // "/view/tasks" (query string stripped)
  std::string query;   // "variant=eager" (without '?')
};

// One accepted connection. Writes are blocking; a failed write means the
// client went away, which long-lived event streams use to stop.
class HttpConnection {
 public:
  explicit HttpConnection(int socket_descriptor);
  ~HttpConnection();
  HttpConnection(const HttpConnection&) = delete;
  HttpConnection& operator=(const HttpConnection&) = delete;

  bool send_response(int status_code, const std::string& content_type,
                     const std::string& response_body);
  bool begin_event_stream();
  bool send_event(const std::string& event_name, const std::string& event_data);
  bool send_keepalive_comment();

 private:
  bool write_all(const std::string& payload);
  int socket_descriptor_;
};

class HttpHandler {
 public:
  virtual ~HttpHandler() = default;
  // Called on a per-connection thread; must be thread-safe.
  virtual void handle(const HttpRequest& request, HttpConnection& connection) = 0;
};

class HttpServer {
 public:
  explicit HttpServer(HttpHandler& request_handler);
  ~HttpServer();
  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  // Binds 127.0.0.1:`requested_port` (0 = any free port) and starts
  // accepting. Returns the bound port, or 0 on failure.
  std::uint16_t start(std::uint16_t requested_port);
  void stop();

 private:
  void accept_loop();

  HttpHandler& request_handler_;
  int listen_descriptor_ = -1;
  std::atomic<bool> is_running_{false};
  std::thread accept_thread_;
};

// Parses "GET /path?query HTTP/1.1"; false when the line is malformed.
bool parse_request_line(const std::string& request_line, HttpRequest& request);

} // namespace mlx_perf
