#include "live_view_page.h"

#include "report_assets.h"
#include "text_format.h"

namespace mlx_plan {
namespace {

std::string_view live_view_stylesheet() {
  return R"CSS(
.tabs { display: flex; gap: 4px; padding: 0 24px; border-bottom: 1px solid var(--rule); }
.tabs button { background: none; border: 0; border-bottom: 2px solid transparent;
  color: var(--muted); font: inherit; font-weight: 600; padding: 10px 12px; cursor: pointer; }
.tabs button[aria-selected="true"] { color: var(--ink); border-bottom-color: var(--slice-queued); }
.toolbar { display: flex; align-items: center; gap: 12px; padding: 8px 24px; color: var(--muted); }
.toolbar button { font: inherit; padding: 4px 12px; border-radius: 4px; cursor: pointer;
  border: 1px solid var(--rule); background: var(--panel); color: var(--ink); }
.variant-heading { margin: 18px 24px 0; font-size: 16px; border: 0; }
#view { padding-bottom: 32px; }
)CSS";
}

// Tabs reload only when views_version changes, so a status tick never
// throws away a table the reader just sorted. The memory tab also polls,
// because its live chart is drawn from samples that do not bump versions.
std::string_view live_view_script() {
  return R"JS(
var currentTab = "job";
var shownViewsVersion = -1;
var view = document.getElementById("view");
function showTab(tab) {
  currentTab = tab;
  document.querySelectorAll(".tabs button").forEach(function (button) {
    button.setAttribute("aria-selected", button.dataset.tab === tab ? "true" : "false");
  });
  fetch("/view/" + tab).then(function (response) { return response.text(); })
    .then(function (fragment) { view.innerHTML = fragment; enableSortableTables(view); });
}
function refreshStatus() {
  fetch("/api/status").then(function (response) { return response.json(); })
    .then(function (status) {
      document.getElementById("status").textContent = status.status;
      if (status.views_version !== shownViewsVersion) {
        shownViewsVersion = status.views_version;
        showTab(currentTab);
      }
    });
}
document.querySelectorAll(".tabs button").forEach(function (button) {
  button.addEventListener("click", function () { showTab(button.dataset.tab); });
});
document.getElementById("rerun").addEventListener("click", function () {
  fetch("/api/run", { method: "POST" });
});
new EventSource("/events").addEventListener("update", refreshStatus);
setInterval(function () { if (currentTab === "memory") { showTab("memory"); } }, 2000);
refreshStatus();
)JS";
}

} // namespace

std::string build_live_view_page(const std::string& job_title) {
  std::string page_html;
  page_html += "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
               "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
               "<title>MLX live plan</title><style>";
  page_html += report_stylesheet();
  page_html += live_view_stylesheet();
  page_html += "</style></head><body><header><h1>" + escape_html_text(job_title) +
      "</h1><p>mlx-visual-plan live — what MLX actually did, and what it was"
      " waiting for</p></header><nav class=\"tabs\" role=\"tablist\">"
      "<button data-tab=\"job\" aria-selected=\"true\">Job</button>"
      "<button data-tab=\"stages\">Stages</button>"
      "<button data-tab=\"tasks\">Tasks</button>"
      "<button data-tab=\"memory\">Memory</button></nav>"
      "<div class=\"toolbar\"><button id=\"rerun\">Re-run</button>"
      "<span id=\"status\">connecting…</span></div><main id=\"view\"></main><script>";
  page_html += report_sort_function();
  page_html += live_view_script();
  page_html += "</script></body></html>\n";
  return page_html;
}

} // namespace mlx_plan
