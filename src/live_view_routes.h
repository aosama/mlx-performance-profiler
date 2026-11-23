#pragma once

// Live view routes: one concern only — map HTTP paths to answers from the
// LiveViewStore. Runs on HTTP threads, so it reads only the store (plain
// data) and never touches MLX.
//
//   GET  /             the tabbed shell page
//   GET  /view/<tab>   job | stages | tasks | memory  (HTML fragment)
//   GET  /api/status   {"status","version","views_version"}
//   POST /api/run      ask the main thread for a fresh measurement (202)
//   GET  /events       Server-Sent Events: "update" whenever version moves

#include "http_server.h"
#include "live_view_store.h"

#include <atomic>
#include <string>

namespace mlx_plan {

class LiveViewRoutes final : public HttpHandler {
 public:
  LiveViewRoutes(LiveViewStore& live_view_store, std::string job_title);
  void handle(const HttpRequest& request, HttpConnection& connection) override;
  // Ends open event streams promptly (they otherwise idle on keep-alives).
  void close_event_streams();

 private:
  void serve_view(const std::string& tab_name, HttpConnection& connection) const;
  void serve_status(HttpConnection& connection) const;
  void serve_event_stream(HttpConnection& connection) const;

  LiveViewStore& live_view_store_;
  std::string shell_page_html_;
  std::atomic<bool> are_event_streams_closing_{false};
};

} // namespace mlx_plan
