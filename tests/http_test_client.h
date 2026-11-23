#pragma once

// A raw-socket HTTP client for tests: sends bytes verbatim (so malformed and
// oversized requests can be tested) and reads until the server closes or a
// receive timeout expires (so event streams can be sampled).

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cstdint>
#include <string>

namespace http_test_client {

inline std::string send_raw_request(std::uint16_t port, const std::string& raw_request,
                            int receive_timeout_milliseconds = 3000) {
  const int client_descriptor = socket(AF_INET, SOCK_STREAM, 0);
  timeval receive_timeout{receive_timeout_milliseconds / 1000,
                          (receive_timeout_milliseconds % 1000) * 1000};
  setsockopt(client_descriptor, SOL_SOCKET, SO_RCVTIMEO, &receive_timeout,
             sizeof(receive_timeout));
  sockaddr_in server_address{};
  server_address.sin_family = AF_INET;
  server_address.sin_port = htons(port);
  server_address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (connect(client_descriptor, reinterpret_cast<sockaddr*>(&server_address),
              sizeof(server_address)) != 0) {
    close(client_descriptor);
    return "";
  }
  send(client_descriptor, raw_request.data(), raw_request.size(), 0);
  std::string response_text;
  char receive_bytes[4096];
  ssize_t received_count = 0;
  while ((received_count = recv(client_descriptor, receive_bytes,
                                sizeof(receive_bytes), 0)) > 0) {
    response_text.append(receive_bytes, static_cast<std::size_t>(received_count));
  }
  close(client_descriptor);
  return response_text;
}

inline std::string get(std::uint16_t port, const std::string& path) {
  return send_raw_request(port, "GET " + path + " HTTP/1.1\r\nHost: localhost\r\n\r\n");
}

inline std::string post(std::uint16_t port, const std::string& path) {
  return send_raw_request(port, "POST " + path +
                            " HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n");
}

} // namespace http_test_client
