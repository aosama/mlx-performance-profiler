#include "http_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <functional>
#include <sstream>

namespace mlx_plan {
namespace {

constexpr std::size_t MAX_REQUEST_HEAD_BYTES = 8192;
constexpr int LISTEN_BACKLOG = 16;

const char* status_text(int status_code) {
  switch (status_code) {
    case 200: return "OK";
    case 202: return "Accepted";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 431: return "Request Header Fields Too Large";
    default: return "Internal Server Error";
  }
}

// Reads until the blank line that ends the request head. Returns false when
// the client closed early or the head exceeds the limit (`is_oversized`).
bool read_request_head(int client_descriptor, std::string& request_head,
                       bool& is_oversized) {
  char receive_bytes[1024];
  while (request_head.find("\r\n\r\n") == std::string::npos) {
    if (request_head.size() > MAX_REQUEST_HEAD_BYTES) {
      is_oversized = true;
      return false;
    }
    const ssize_t received_count =
        recv(client_descriptor, receive_bytes, sizeof(receive_bytes), 0);
    if (received_count <= 0) {
      return false;
    }
    request_head.append(receive_bytes, static_cast<std::size_t>(received_count));
  }
  return true;
}

// A free function, not a member: detached connection threads must never
// touch the HttpServer, which may be destroyed while an event stream is
// still draining. They only need the handler, which outlives them.
void serve_connection(HttpHandler& request_handler, int client_descriptor) {
  HttpConnection connection(client_descriptor);
  std::string request_head;
  bool is_oversized = false;
  if (!read_request_head(client_descriptor, request_head, is_oversized)) {
    if (is_oversized) {
      connection.send_response(431, "text/plain", "request head too large\n");
    }
    return;
  }
  HttpRequest request;
  if (!parse_request_line(request_head.substr(0, request_head.find("\r\n")), request)) {
    connection.send_response(400, "text/plain", "malformed request line\n");
    return;
  }
  request_handler.handle(request, connection);
}

} // namespace

bool parse_request_line(const std::string& request_line, HttpRequest& request) {
  std::istringstream line_reader(request_line);
  std::string target;
  std::string protocol;
  if (!(line_reader >> request.method >> target >> protocol) ||
      protocol.rfind("HTTP/", 0) != 0 || target.empty() || target[0] != '/') {
    return false;
  }
  const std::size_t query_start = target.find('?');
  request.path = target.substr(0, query_start);
  request.query =
      query_start == std::string::npos ? "" : target.substr(query_start + 1);
  return true;
}

HttpConnection::HttpConnection(int socket_descriptor)
    : socket_descriptor_(socket_descriptor) {}

HttpConnection::~HttpConnection() { close(socket_descriptor_); }

bool HttpConnection::write_all(const std::string& payload) {
  std::size_t written_total = 0;
  while (written_total < payload.size()) {
    // MSG_NOSIGNAL is not on macOS; SO_NOSIGPIPE is set on the socket instead.
    const ssize_t written_count = send(socket_descriptor_, payload.data() + written_total,
                                       payload.size() - written_total, 0);
    if (written_count <= 0) {
      return false;
    }
    written_total += static_cast<std::size_t>(written_count);
  }
  return true;
}

bool HttpConnection::send_response(int status_code, const std::string& content_type,
                                   const std::string& response_body) {
  std::ostringstream response_text;
  response_text << "HTTP/1.1 " << status_code << " " << status_text(status_code)
                << "\r\nContent-Type: " << content_type
                << "\r\nContent-Length: " << response_body.size()
                << "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n"
                << response_body;
  return write_all(response_text.str());
}

bool HttpConnection::begin_event_stream() {
  return write_all("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                   "Cache-Control: no-store\r\nConnection: keep-alive\r\n\r\n");
}

bool HttpConnection::send_event(const std::string& event_name,
                                const std::string& event_data) {
  return write_all("event: " + event_name + "\ndata: " + event_data + "\n\n");
}

bool HttpConnection::send_keepalive_comment() { return write_all(": keepalive\n\n"); }

HttpServer::HttpServer(HttpHandler& request_handler)
    : request_handler_(request_handler) {}

HttpServer::~HttpServer() { stop(); }

std::uint16_t HttpServer::start(std::uint16_t requested_port) {
  listen_descriptor_ = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_descriptor_ < 0) {
    return 0;
  }
  const int enable_option = 1;
  setsockopt(listen_descriptor_, SOL_SOCKET, SO_REUSEADDR, &enable_option,
             sizeof(enable_option));
  sockaddr_in listen_address{};
  listen_address.sin_family = AF_INET;
  listen_address.sin_port = htons(requested_port);
  // Loopback only: this is a local developer view, never a network service.
  listen_address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (bind(listen_descriptor_, reinterpret_cast<sockaddr*>(&listen_address),
           sizeof(listen_address)) != 0 ||
      listen(listen_descriptor_, LISTEN_BACKLOG) != 0) {
    close(listen_descriptor_);
    listen_descriptor_ = -1;
    return 0;
  }
  socklen_t address_length = sizeof(listen_address);
  getsockname(listen_descriptor_, reinterpret_cast<sockaddr*>(&listen_address),
              &address_length);
  is_running_ = true;
  accept_thread_ = std::thread(&HttpServer::accept_loop, this);
  return ntohs(listen_address.sin_port);
}

void HttpServer::stop() {
  if (!is_running_.exchange(false)) {
    return;
  }
  // shutdown() wakes the blocked accept() so the loop can observe the flag.
  shutdown(listen_descriptor_, SHUT_RDWR);
  close(listen_descriptor_);
  if (accept_thread_.joinable()) {
    accept_thread_.join();
  }
}

void HttpServer::accept_loop() {
  while (is_running_) {
    const int client_descriptor = accept(listen_descriptor_, nullptr, nullptr);
    if (client_descriptor < 0) {
      continue;
    }
    const int enable_option = 1;
    // A client closing mid-stream must surface as a failed send, not SIGPIPE.
    setsockopt(client_descriptor, SOL_SOCKET, SO_NOSIGPIPE, &enable_option,
               sizeof(enable_option));
    // One thread per connection: the event stream is long-lived, and a local
    // single-user view never has more than a handful of connections.
    std::thread(serve_connection, std::ref(request_handler_), client_descriptor)
        .detach();
  }
}

} // namespace mlx_plan
