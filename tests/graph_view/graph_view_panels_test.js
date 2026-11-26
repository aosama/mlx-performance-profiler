// Unit tests for src/graph_view/graph_view_panels.js: the hover card,
// details panel and legend HTML. Pure strings, no browser.
//
// Run: node --test tests/graph_view/

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

for (const moduleName of ["graph_view_model.js", "graph_view_style.js", "graph_view_panels.js"]) {
  const sourcePath = path.join(__dirname, "..", "..", "src", "graph_view", moduleName);
  vm.runInThisContext(fs.readFileSync(sourcePath, "utf8"), { filename: sourcePath });
}
const graphView = globalThis.mlxPerfGraphView;
const THEME = { "--slice-cpu": "#1", "--slice-queued": "#2", "--slice-stalled": "#3",
                "--slice-scheduling": "#4", "--slice-compute": "#5", "--slice-result": "#6",
                compute: "#5", wait_on_other_stream: "#3" };

function timedNode(overrides) {
  return Object.assign({ position: 0, label: "A", op_name: "MatMul", kernel_name: null, stream_index: 0,
    shape: [2, 2], dtype: "float32", output_bytes: 16, input_positions: [], serial_duration_us: 9,
    classification: "compute", wall_us: 10, compute_us: 8, wait_on_host_us: 2 }, overrides);
}

function describeNode(documentNodes, stages) {
  const graphModel = graphView.buildGraphModel({ nodes: documentNodes, stages: stages || [] });
  return graphView.describeElementHtml(graphModel, THEME, { kind: "node", position: 0 });
}

// --- positive ---

test("should_describe_a_timed_node_with_its_slices_and_wall", () => {
  const cardHtml = describeNode([timedNode()]);
  assert.match(cardHtml, /<h4>MatMul <small>A<\/small><\/h4>/);
  assert.match(cardHtml, /mostly compute/);
  assert.match(cardHtml, /width:80\.00%;background:#5/);
  assert.match(cardHtml, /<td>wall<\/td><td>10\.0 µs<\/td>/);
});

test("should_show_a_stage_timed_node_with_its_stages_breakdown", () => {
  const stageRecord = { stage_position: 0, stage_name: "layer 0", stream_index: 0, node_positions: [0],
                        classification: "wait_on_other_stream", wall_us: 50, wait_on_other_stream_us: 50 };
  const cardHtml = describeNode([timedNode({ wall_us: null, compute_us: null, classification: null })], [stageRecord]);
  assert.match(cardHtml, /mostly stalled on another stream \(its stage\)/);
  assert.match(cardHtml, /Timed as part of its stage/);
  assert.match(cardHtml, /<td>wall<\/td><td>50\.0 µs<\/td>/);
});

test("should_list_every_class_in_the_legend_when_colouring_by_time", () => {
  const legendHtml = graphView.renderLegendHtml(THEME, "time",
    { shownClassifications: ["compute"], streamIndices: [0], isCostRunAlone: false });
  assert.match(legendHtml, /mostly compute/);
  assert.match(legendHtml, /mostly stalled on another stream/);
});

// --- edge / boundary ---

test("should_leave_out_the_slice_bar_when_a_node_has_no_wall_time", () => {
  assert.doesNotMatch(describeNode([timedNode({ wall_us: 0 })]), /gv-slice-bar/);
});

test("should_explain_cost_legend_entries_for_nodes_run_alone", () => {
  const legendHtml = graphView.renderLegendHtml(THEME, "time",
    { shownClassifications: [], streamIndices: [0], isCostRunAlone: true });
  assert.match(legendHtml, /most expensive \(each node run alone\)/);
});

// --- negative ---

test("should_escape_markup_in_an_author_chosen_kernel_name", () => {
  const cardHtml = describeNode([timedNode({ op_name: "CustomKernel", kernel_name: "<img src=x onerror=alert(1)>" })]);
  assert.doesNotMatch(cardHtml, /<img/);
  assert.match(cardHtml, /&lt;img src=x onerror=alert\(1\)&gt;/);
});

test("should_escape_every_html_special_character", () => {
  assert.equal(graphView.escapeHtml("<a href=\"x\">&'</a>"), "&lt;a href=&quot;x&quot;&gt;&amp;&#39;&lt;/a&gt;");
});
