#include "report_assets.h"

#include <string>

namespace mlx_perf {

std::string_view report_stylesheet() {
  // Slice colors follow Spark's legend semantics (computing green, stalled
  // red) and are tokens so light and dark themes stay in sync.
  return R"CSS(
:root {
  --page: #ffffff; --ink: #1d2430; --muted: #5d6b7c; --rule: #d9dfe7;
  --panel: #f5f7fa; --header: #16212e; --header-ink: #ffffff;
  --slice-cpu: #8b95a7; --slice-queued: #6b9bf2; --slice-stalled: #e5484d;
  --slice-scheduling: #f0a020; --slice-compute: #2f9e64; --slice-result: #c9ced8;
  --lane: #eef2f6;
}
@media (prefers-color-scheme: dark) {
  :root {
    --page: #11161d; --ink: #e4e9f0; --muted: #94a3b5; --rule: #2a3441;
    --panel: #18202a; --header: #0b1016; --header-ink: #e4e9f0;
    --slice-cpu: #77829a; --slice-queued: #5a8ae0; --slice-stalled: #f0565b;
    --slice-scheduling: #e8a43a; --slice-compute: #3fb77a; --slice-result: #4a5363;
    --lane: #1b2530;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 13px/1.45 -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif; }
header { background: var(--header); color: var(--header-ink); padding: 14px 24px; }
header h1 { margin: 0; font-size: 19px; font-weight: 600; }
header p { margin: 2px 0 0; opacity: .7; font-size: 12px; }
nav { display: flex; gap: 18px; padding: 8px 24px; border-bottom: 1px solid var(--rule); }
nav a { color: var(--muted); text-decoration: none; font-weight: 500; }
nav a:hover { color: var(--ink); }
section { padding: 14px 24px 6px; }
h2 { font-size: 15px; margin: 6px 0 10px; padding-bottom: 6px; border-bottom: 1px solid var(--rule); }
.note { color: var(--muted); font-size: 12px; max-width: 980px; }
.verdict { font-size: 15px; font-weight: 600; max-width: 980px; margin: 4px 0 12px;
  background: var(--panel); border: 1px solid var(--rule); border-left: 4px solid var(--slice-compute);
  border-radius: 6px; padding: 10px 14px; }
.verdict.slower { border-left-color: var(--slice-stalled); }
.verdict.within-spread { border-left-color: var(--muted); }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 10px; }
.card { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 10px 14px; }
.card-value { font-size: 21px; font-weight: 650; font-variant-numeric: tabular-nums; }
.card-label { font-size: 12px; font-weight: 600; margin-top: 2px; }
.card-detail { font-size: 11px; color: var(--muted); margin-top: 3px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; margin: 6px 0 10px; font-size: 12px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 12px; height: 12px; border-radius: 2px; display: inline-block; }
.slice-cpu { background: var(--slice-cpu); fill: var(--slice-cpu); }
.slice-queued { background: var(--slice-queued); fill: var(--slice-queued); }
.slice-stalled { background: var(--slice-stalled); fill: var(--slice-stalled); }
.slice-scheduling { background: var(--slice-scheduling); fill: var(--slice-scheduling); }
.slice-compute { background: var(--slice-compute); fill: var(--slice-compute); }
.slice-result { background: var(--slice-result); fill: var(--slice-result); }
.stacked-bar { display: flex; height: 14px; min-width: 160px; border-radius: 2px; overflow: hidden; background: var(--lane); }
.stacked-bar > span { display: block; height: 100%; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; font-size: 12px; width: 100%; }
th, td { border-bottom: 1px solid var(--rule); padding: 4px 8px; text-align: left; white-space: nowrap; }
th { background: var(--panel); position: sticky; top: 0; }
table.sortable th { cursor: pointer; user-select: none; }
table.sortable th[data-sort-direction="ascending"]::after { content: " ▲"; }
table.sortable th[data-sort-direction="descending"]::after { content: " ▼"; }
td.number { text-align: right; font-variant-numeric: tabular-nums; }
td.underfills-gpu, .underfills-gpu { color: var(--slice-scheduling); font-weight: 600; }
.class-badge { border-radius: 3px; padding: 1px 6px; font-size: 11px; color: #fff; }
.gantt-wrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 6px; background: var(--panel); }
.gantt text { fill: var(--ink); font-size: 11px; }
.gantt .axis-text { fill: var(--muted); font-size: 10px; }
.gantt .lane { fill: var(--lane); }
.gantt .grid { stroke: var(--rule); stroke-width: 1; }
.gantt .dependency-arrow { stroke: var(--slice-stalled); stroke-width: 1; fill: none; opacity: .55; }
.wired-chart { width: 100%; max-width: 1100px; height: auto; display: block; background: var(--panel);
  border: 1px solid var(--rule); border-radius: 6px; }
.wired-chart .grid { stroke: var(--rule); stroke-width: 1; }
.wired-chart .axis { fill: var(--muted); font-size: 10px; }
.wired-chart polyline { fill: none; stroke-width: 1.6; stroke-linejoin: round; }
.wired-chart .system-wired { stroke: var(--slice-queued); }
.wired-chart .mlx-held { stroke: var(--slice-compute); stroke-dasharray: 5 3; }
.wired-chart .outside-mlx { stroke: var(--slice-scheduling); }
.wired-chart .limit { stroke: var(--slice-stalled); stroke-width: 1.2; stroke-dasharray: 2 3; }
.wired-chart .release { stroke: var(--slice-compute); stroke-width: 1; stroke-dasharray: 3 3; }
.wired-chart .unwired { stroke: var(--slice-queued); stroke-width: 1; stroke-dasharray: 3 3; }
.wired-chart .pending-unwire { fill: var(--slice-stalled); }
.wired-chart .transient { fill: none; stroke: var(--slice-stalled); stroke-width: 1.6; }
.wired-chart .boundary { stroke: var(--muted); stroke-width: 1; }
.wired-legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 11px; color: var(--muted); margin: 6px 0 8px; }
.wired-legend .key::before { content: ""; display: inline-block; width: 14px; height: 3px; margin-right: 5px;
  vertical-align: middle; background: currentColor; }
.wired-legend .system-wired::before { background: var(--slice-queued); }
.wired-legend .mlx-held::before { background: var(--slice-compute); }
.wired-legend .outside-mlx::before { background: var(--slice-scheduling); }
.wired-legend .pending-unwire::before { width: 6px; height: 6px; border-radius: 3px; background: var(--slice-stalled); }
.wired-legend .transient::before { width: 8px; height: 8px; border-radius: 5px; background: none;
  border: 1.6px solid var(--slice-stalled); }
.wired-legend .boundary::before { width: 1px; height: 8px; background: var(--muted); }
table.wired-summary { max-width: 720px; }
.line-compute { stroke: var(--slice-compute); }
.line-scheduling { stroke: var(--slice-scheduling); stroke-dasharray: 4 3; }
.line-result { stroke: var(--slice-result); }
.variant-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 16px; }
.variant h3, section h3 { font-size: 13px; margin: 12px 0 6px; }
.variant small { font-weight: 400; color: var(--muted); }
a { color: var(--slice-queued); }
details summary { cursor: pointer; color: var(--muted); }
pre { background: var(--panel); border: 1px solid var(--rule); padding: 10px; overflow: auto; font-size: 11px; }
)CSS";
}

std::string_view report_sort_function() {
  return R"JS(
function enableSortableTables(root) {
root.querySelectorAll("table.sortable").forEach(function (table) {
  table.querySelectorAll("th").forEach(function (header, columnIndex) {
    header.addEventListener("click", function () {
      var ascending = header.getAttribute("data-sort-direction") !== "ascending";
      table.querySelectorAll("th").forEach(function (other) {
        other.removeAttribute("data-sort-direction");
      });
      header.setAttribute("data-sort-direction", ascending ? "ascending" : "descending");
      var body = table.tBodies[0];
      var rows = Array.prototype.slice.call(body.rows);
      rows.sort(function (left, right) {
        var leftCell = left.cells[columnIndex], rightCell = right.cells[columnIndex];
        var leftValue = leftCell.getAttribute("data-sort-value");
        var rightValue = rightCell.getAttribute("data-sort-value");
        var order;
        if (leftValue !== null && rightValue !== null) {
          order = parseFloat(leftValue) - parseFloat(rightValue);
        } else {
          order = leftCell.textContent.localeCompare(rightCell.textContent);
        }
        return ascending ? order : -order;
      });
      rows.forEach(function (row) { body.appendChild(row); });
    });
  });
});
}
)JS";
}

std::string_view report_sort_script() {
  // Static reports: define the function and apply it to the whole page.
  static const std::string sort_script =
      std::string(report_sort_function()) + "enableSortableTables(document);\n";
  return sort_script;
}

} // namespace mlx_perf
