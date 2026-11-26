// Unit tests for src/graph_view/graph_view_model.js: what the graph view
// draws from a measurement document. Pure data in, elements out; no browser.
//
// Run: node --test tests/graph_view/   (CTest runs it when Node is installed)

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const modelSourcePath = path.join(__dirname, "..", "..", "src", "graph_view", "graph_view_model.js");
vm.runInThisContext(fs.readFileSync(modelSourcePath, "utf8"), { filename: modelSourcePath });
const graphView = globalThis.mlxPerfGraphView;

function nodeRecord(position, overrides) {
  return Object.assign({ position: position, label: "N" + position, op_name: "Add", kernel_name: null,
    stream_index: 0, shape: [4], dtype: "float32", output_bytes: 16, input_positions: [],
    cross_stream_input_positions: [], compute_us: 10, is_compute_gpu_measured: true,
    serial_duration_us: 12, classification: "compute", wall_us: 20 }, overrides);
}

function stageTimedNode(position, inputPositions, serialMicroseconds) {
  return nodeRecord(position, { input_positions: inputPositions, compute_us: null, classification: null,
    is_compute_gpu_measured: null, wall_us: null, serial_duration_us: serialMicroseconds });
}

function stageRecord(stagePosition, nodePositions, computeMicroseconds) {
  return { stage_position: stagePosition, stage_name: "layer " + stagePosition, stream_index: 0,
           node_positions: nodePositions, compute_us: computeMicroseconds, classification: "compute" };
}

function threeStageDocument() {
  return { variant_name: "eager", granularity: "per_stage",
    nodes: [stageTimedNode(0, [], 5), stageTimedNode(1, [0], 30), stageTimedNode(2, [1], 7),
            stageTimedNode(3, [2, 1], 9)],
    stages: [stageRecord(0, [0], 50), stageRecord(1, [1, 2], 200), stageRecord(2, [3], 100)] };
}

function elementsOfGroup(elements, group) {
  return elements.filter((element) => element.group === group);
}

// --- positive ---

test("should_build_one_node_per_primitive_and_one_edge_per_input", () => {
  const graphModel = graphView.buildGraphModel({ nodes: [nodeRecord(0), nodeRecord(1, { input_positions: [0] })] });
  const elements = graphView.buildNodeElements(graphModel, null);
  assert.equal(elementsOfGroup(elements, "nodes").length, 2);
  const edges = elementsOfGroup(elements, "edges");
  assert.deepEqual(edges.map((edge) => [edge.data.source, edge.data.target, edge.data.bytes]), [["n0", "n1", 16]]);
});

test("should_mark_an_edge_read_from_another_stream", () => {
  const graphModel = graphView.buildGraphModel({ nodes: [nodeRecord(0, { stream_index: 1 }),
    nodeRecord(1, { input_positions: [0], cross_stream_input_positions: [0] })] });
  const edge = elementsOfGroup(graphView.buildNodeElements(graphModel, null), "edges")[0];
  assert.equal(edge.data.isCrossStream, true);
});

test("should_caption_a_node_with_its_gpu_time_and_output_size", () => {
  const graphModel = graphView.buildGraphModel({ nodes: [nodeRecord(0, { compute_us: 125.5, output_bytes: 2048 })] });
  assert.equal(graphView.buildNodeElements(graphModel, null)[0].data.caption, "Add\ngpu 125.5 µs · 2.0 KiB");
});

test("should_cost_a_stage_timed_node_by_its_time_run_alone", () => {
  const graphModel = graphView.buildGraphModel(threeStageDocument());
  const node = graphView.buildNodeElements(graphModel, null).find((element) => element.data.id === "n1");
  assert.equal(node.data.cost, 30);
  assert.equal(node.data.costShare, 1);
  assert.match(node.data.caption, /alone 30\.0 µs/);
});

test("should_join_stages_with_the_bytes_that_cross_between_them", () => {
  const graphModel = graphView.buildGraphModel(threeStageDocument());
  const stageEdges = elementsOfGroup(graphView.buildStageElements(graphModel), "edges");
  // 1→2 inside stage 1 is not a stage edge; 2→3 and 1→3 both cross 1→2.
  assert.deepEqual(stageEdges.map((edge) => [edge.data.source, edge.data.target, edge.data.bytes]),
                   [["s0", "s1", 16], ["s1", "s2", 32]]);
});

test("should_open_a_stage_with_one_ghost_per_neighbouring_stage", () => {
  const graphModel = graphView.buildGraphModel(threeStageDocument());
  const elements = graphView.buildStageDrillDownElements(graphModel, 1);
  const nodeIds = elementsOfGroup(elements, "nodes").map((element) => element.data.id).sort();
  assert.deepEqual(nodeIds, ["from0", "n1", "n2", "to2"]);
  const ghostEdges = elementsOfGroup(elements, "edges").filter((edge) => edge.data.isGhostEdge);
  assert.deepEqual(ghostEdges.map((edge) => edge.data.id).sort(), ["gfrom0_n1", "gn1_to2", "gn2_to2"]);
});

test("should_find_the_most_expensive_dependency_chain", () => {
  // a → b → d costs 1 + 5 + 1; a → c → d costs 1 + 2 + 1.
  const chain = graphView.findMostExpensiveChain(
    [{ id: "a", cost: 1 }, { id: "b", cost: 5 }, { id: "c", cost: 2 }, { id: "d", cost: 1 }],
    { b: ["a"], c: ["a"], d: ["b", "c"] });
  assert.deepEqual(chain.nodeIds, ["a", "b", "d"]);
  assert.equal(chain.totalMicroseconds, 7);
});

test("should_recognise_a_chain_of_stages", () => {
  const graphModel = graphView.buildGraphModel(threeStageDocument());
  assert.equal(graphView.isSimpleStageChain(graphView.buildStageElements(graphModel)), true);
});

test("should_lay_a_chain_out_as_a_snake", () => {
  assert.deepEqual(graphView.snakePositions(5, 2, 100, 50),
                   [{ x: 0, y: 0 }, { x: 100, y: 0 }, { x: 100, y: 50 }, { x: 0, y: 50 }, { x: 0, y: 100 }]);
});

function launchRecord(threadgroupCount, metalFunctionName) {
  return { metal_function_name: metalFunctionName || "custom_kernel_attention", threadgroup_count: threadgroupCount,
           threadgroup_thread_count: 32 };
}

test("should_mark_a_node_whose_launch_leaves_gpu_cores_idle", () => {
  const graphModel = graphView.buildGraphModel({ job: { gpu: { core_count: 16 } },
    nodes: [nodeRecord(0, { kernel_dispatches: [launchRecord(4)] })] });
  const node = graphView.buildNodeElements(graphModel, null)[0];
  assert.equal(node.data.cannotFillGpu, true);
  assert.equal(node.data.lowestCoverage, 0.25);
  assert.match(node.data.caption, /\n⚠ 4 of 16 cores$/);
  assert.equal(node.data.height, 56);
});

test("should_count_a_stages_launches_that_leave_cores_idle", () => {
  const document = threeStageDocument();
  document.job = { gpu: { core_count: 16 } };
  document.nodes[1].kernel_dispatches = [launchRecord(4), launchRecord(8)];
  document.nodes[2].kernel_dispatches = [launchRecord(512)];
  const stage = graphView.buildStageElements(graphView.buildGraphModel(document)).find((element) => element.data.id === "s1");
  assert.match(stage.data.caption, /⚠ 2 launches leave cores idle/);
  assert.equal(stage.data.lowestCoverage, 0.25);
});

test("should_find_a_node_by_the_metal_function_it_launched", () => {
  const graphModel = graphView.buildGraphModel({ nodes: [nodeRecord(0, {
    kernel_dispatches: [launchRecord(64, "gemv_t_bfloat16_bm8")] })] });
  assert.match(graphView.buildNodeElements(graphModel, null)[0].data.searchText, /gemv_t_bfloat16_bm8/);
});

// --- edge / boundary ---

test("should_fill_the_gpu_with_one_threadgroup_per_core", () => {
  const launches = graphView.describeNodeLaunches(nodeRecord(0, { kernel_dispatches: [launchRecord(16)] }), 16);
  assert.equal(launches.underfillingLaunchCount, 0);
  assert.equal(launches.lowestCoverage, 1);
});

test("should_describe_no_launches_for_a_view_or_an_old_file", () => {
  assert.equal(graphView.describeNodeLaunches(nodeRecord(0, { kernel_dispatches: [] }), 16), null);
  assert.equal(graphView.describeNodeLaunches(nodeRecord(0), 16), null);
  const graphModel = graphView.buildGraphModel({ nodes: [nodeRecord(0)] });
  assert.equal(graphModel.gpuCoreCount, 0);
  assert.equal(graphView.buildNodeElements(graphModel, null)[0].data.height, 42);
});

test("should_draw_nothing_for_an_empty_document", () => {
  const graphModel = graphView.buildGraphModel({});
  assert.equal(graphModel.isStageTimed, false);
  assert.deepEqual(graphView.buildNodeElements(graphModel, null), []);
  assert.deepEqual(graphView.findMostExpensiveChain([], {}), { nodeIds: [], totalMicroseconds: 0 });
});

test("should_place_a_snake_in_one_column_when_only_one_fits", () => {
  assert.deepEqual(graphView.snakePositions(3, 0, 100, 50), [{ x: 0, y: 0 }, { x: 0, y: 50 }, { x: 0, y: 100 }]);
});

test("should_use_one_row_when_columns_outnumber_tiles", () => {
  assert.deepEqual(graphView.snakePositions(2, 10, 100, 50), [{ x: 0, y: 0 }, { x: 100, y: 0 }]);
});

test("should_switch_units_at_one_millisecond_and_one_kibibyte", () => {
  assert.equal(graphView.formatMicroseconds(999.9), "999.9 µs");
  assert.equal(graphView.formatMicroseconds(1000), "1.00 ms");
  assert.equal(graphView.formatBytes(1023), "1023 B");
  assert.equal(graphView.formatBytes(1024), "1.0 KiB");
});

test("should_shorten_a_long_fused_kernel_name", () => {
  const shortened = graphView.shortenOpName("CompiledExpBroadcastBroadcastAddAsTypeBroadcastLog");
  assert.equal(shortened.length, 26);
  assert.ok(shortened.endsWith("…"));
  assert.equal(graphView.shortenOpName("Add"), "Add");
});

// --- negative ---

test("should_claim_no_idle_cores_when_the_core_count_is_unknown", () => {
  const graphModel = graphView.buildGraphModel({ nodes: [nodeRecord(0, { kernel_dispatches: [launchRecord(4)] })] });
  const node = graphView.buildNodeElements(graphModel, null)[0];
  assert.equal(node.data.cannotFillGpu, false);
  assert.equal(node.data.lowestCoverage, -1);
  assert.doesNotMatch(node.data.caption, /⚠/);
});

test("should_not_call_a_branching_stage_graph_a_chain", () => {
  const stageElements = [
    { group: "nodes", data: { id: "s0" } }, { group: "nodes", data: { id: "s1" } }, { group: "nodes", data: { id: "s2" } },
    { group: "edges", data: { source: "s0", target: "s1" } }, { group: "edges", data: { source: "s0", target: "s2" } }];
  assert.equal(graphView.isSimpleStageChain(stageElements), false);
  assert.equal(graphView.isSimpleStageChain([{ group: "nodes", data: { id: "s0" } }]), false);
});

test("should_ignore_an_input_that_names_no_executed_node", () => {
  const graphModel = graphView.buildGraphModel({ nodes: [nodeRecord(0, { input_positions: [7] })] });
  assert.equal(elementsOfGroup(graphView.buildNodeElements(graphModel, null), "edges").length, 0);
});

test("should_draw_one_edge_when_a_node_reads_the_same_input_twice", () => {
  const graphModel = graphView.buildGraphModel({ nodes: [nodeRecord(0), nodeRecord(1, { input_positions: [0, 0] })] });
  assert.equal(elementsOfGroup(graphView.buildNodeElements(graphModel, null), "edges").length, 1);
});

test("should_show_unmeasured_values_as_not_available_rather_than_zero", () => {
  assert.equal(graphView.formatMicroseconds(null), "n/a");
  assert.equal(graphView.formatMicroseconds(NaN), "n/a");
  const cost = graphView.describeNodeCost(nodeRecord(0, { compute_us: null, serial_duration_us: null }));
  assert.deepEqual(cost, { microseconds: 0, prefix: "" });
});
