// Unit tests for src/graph_view/graph_view_style.js: which colour a node
// gets. The theme is passed in, so no browser is needed.
//
// Run: node --test tests/graph_view/

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

for (const moduleName of ["graph_view_model.js", "graph_view_style.js"]) {
  const sourcePath = path.join(__dirname, "..", "..", "src", "graph_view", moduleName);
  vm.runInThisContext(fs.readFileSync(sourcePath, "utf8"), { filename: sourcePath });
}
const graphView = globalThis.mlxPerfGraphView;
const THEME = { muted: "#94a3b5", compute: "#3fb77a", wait_on_other_stream: "#f0565b" };

// --- positive ---

test("should_colour_a_node_by_where_its_time_went", () => {
  assert.equal(graphView.nodeColour(THEME, "time", { kind: "node", classification: "wait_on_other_stream" }), "#f0565b");
});

test("should_colour_a_node_by_its_stream", () => {
  assert.equal(graphView.nodeColour(THEME, "stream", { kind: "node", streamIndex: 1 }), graphView.STREAM_COLOURS[1]);
});

test("should_span_the_heat_ramp_from_cheapest_to_most_expensive", () => {
  assert.equal(graphView.heatColour(0), graphView.HEAT_COLOURS[0]);
  assert.equal(graphView.heatColour(1), graphView.HEAT_COLOURS[graphView.HEAT_COLOURS.length - 1]);
});

test("should_build_a_stylesheet_with_node_edge_and_highlight_rules", () => {
  const selectors = graphView.buildStylesheet(THEME, "time").map((rule) => rule.selector);
  for (const selector of ["node", "edge", "edge[?isCrossStream]", "node.hot", "edge.hot", ".faded"]) {
    assert.ok(selectors.includes(selector), selector);
  }
});

test("should_colour_coverage_from_stalled_red_to_compute_green", () => {
  assert.equal(graphView.coverageColour(THEME, 0), THEME.wait_on_other_stream);
  assert.equal(graphView.coverageColour(THEME, 1), THEME.compute);
  assert.equal(graphView.nodeColour(THEME, "coverage", { kind: "node", lowestCoverage: 1 }), THEME.compute);
});

// --- edge / boundary ---

test("should_fall_back_to_cost_for_a_node_timed_only_as_part_of_its_stage", () => {
  const colour = graphView.nodeColour(THEME, "time", { kind: "node", classification: null, costShare: 1 });
  assert.equal(colour, graphView.HEAT_COLOURS[graphView.HEAT_COLOURS.length - 1]);
});

test("should_wrap_stream_colours_past_the_palette", () => {
  const paletteSize = graphView.STREAM_COLOURS.length;
  assert.equal(graphView.nodeColour(THEME, "stream", { kind: "node", streamIndex: paletteSize }),
               graphView.STREAM_COLOURS[0]);
});

// --- negative ---

test("should_colour_a_node_without_launches_muted_in_coverage_mode", () => {
  assert.equal(graphView.coverageColour(THEME, -1), THEME.muted);
  assert.equal(graphView.coverageColour(THEME, undefined), THEME.muted);
});

test("should_clamp_a_cost_share_outside_zero_to_one", () => {
  assert.equal(graphView.heatColour(-3), graphView.HEAT_COLOURS[0]);
  assert.equal(graphView.heatColour(7), graphView.HEAT_COLOURS[graphView.HEAT_COLOURS.length - 1]);
  assert.equal(graphView.heatColour(undefined), graphView.HEAT_COLOURS[0]);
});

test("should_draw_a_ghost_in_the_muted_colour_whatever_the_mode", () => {
  for (const colourMode of ["time", "heat", "stream"]) {
    assert.equal(graphView.nodeColour(THEME, colourMode, { kind: "ghost", streamIndex: 2 }), THEME.muted);
  }
});
