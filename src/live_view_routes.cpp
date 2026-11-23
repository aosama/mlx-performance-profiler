#include "live_view_routes.h"

#include "live_view_page.h"
#include "report_memory_section.h"
#include "text_format.h"

#include <chrono>

namespace mlx_plan {
namespace {

constexpr auto EVENT_STREAM_KEEPALIVE = std::chrono::milliseconds(15000);
constexpr const char* HTML_CONTENT_TYPE = "text/html; charset=utf-8";

} // namespace

LiveViewRoutes::LiveViewRoutes(LiveViewStore& live_view_store, std::string job_title)
    : live_view_store_(live_view_store),
      shell_page_html_(build_live_view_page(job_title)) {}

void LiveViewRoutes::handle(const HttpRequest& request, HttpConnection& connection) {
  const bool is_get = request.method == "GET";
  if (request.path == "/api/run") {
    if (request.method != "POST") {
      connection.send_response(405, "text/plain", "use POST\n");
      return;
    }
    live_view_store_.request_run();
    connection.send_response(202, "application/json", "{\"accepted\":true}");
    return;
  }
  if (!is_get) {
    connection.send_response(405, "text/plain", "method not allowed\n");
  } else if (request.path == "/") {
    connection.send_response(200, HTML_CONTENT_TYPE, shell_page_html_);
  } else if (request.path.rfind("/view/", 0) == 0) {
    serve_view(request.path.substr(6), connection);
  } else if (request.path == "/api/status") {
    serve_status(connection);
  } else if (request.path == "/events") {
    serve_event_stream(connection);
  } else {
    connection.send_response(404, "text/plain", "not found\n");
  }
}

void LiveViewRoutes::close_event_streams() { are_event_streams_closing_ = true; }

void LiveViewRoutes::serve_view(const std::string& tab_name,
                                HttpConnection& connection) const {
  const RenderedViews rendered_views = live_view_store_.views();
  const std::string waiting_note =
      "<section><p class=\"note\">Measuring… this tab fills in when the first"
      " run completes.</p></section>";
  const std::string* tab_html = nullptr;
  if (tab_name == "job") tab_html = &rendered_views.job_html;
  if (tab_name == "stages") tab_html = &rendered_views.stages_html;
  if (tab_name == "tasks") tab_html = &rendered_views.tasks_html;
  if (tab_name == "memory") {
    // The live chart is drawn per request from the latest samples.
    connection.send_response(
        200, HTML_CONTENT_TYPE,
        "<section><h2>Process memory (live, 1 Hz)</h2>" +
            render_memory_timeline(live_view_store_.memory_samples()) +
            "</section>" +
            (rendered_views.memory_html.empty() ? waiting_note
                                                : rendered_views.memory_html));
    return;
  }
  if (tab_html == nullptr) {
    connection.send_response(404, "text/plain", "unknown tab\n");
    return;
  }
  connection.send_response(200, HTML_CONTENT_TYPE,
                           tab_html->empty() ? waiting_note : *tab_html);
}

void LiveViewRoutes::serve_status(HttpConnection& connection) const {
  connection.send_response(
      200, "application/json",
      "{\"status\":\"" + escape_json_text(live_view_store_.status()) +
          "\",\"version\":" + std::to_string(live_view_store_.version()) +
          ",\"views_version\":" + std::to_string(live_view_store_.views_version()) +
          "}");
}

// Holds the connection open and pushes an "update" each time the store's
// version moves; a keep-alive comment every 15 s detects departed clients.
void LiveViewRoutes::serve_event_stream(HttpConnection& connection) const {
  if (!connection.begin_event_stream()) {
    return;
  }
  std::uint64_t seen_version = live_view_store_.version();
  if (!connection.send_event("update", std::to_string(seen_version))) {
    return;
  }
  while (!are_event_streams_closing_) {
    const std::uint64_t current_version =
        live_view_store_.wait_for_version_after(seen_version, EVENT_STREAM_KEEPALIVE);
    const bool is_still_connected =
        current_version > seen_version
            ? connection.send_event("update", std::to_string(current_version))
            : connection.send_keepalive_comment();
    if (!is_still_connected) {
      return;
    }
    seen_version = current_version;
  }
}

} // namespace mlx_plan
