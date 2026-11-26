/*
 * mlx-performance-profiler graph view: the model.
 *
 * Turns a measurement document (schema version 1, docs/measurement-file.md)
 * into the nodes and edges the view draws. Pure: no DOM and no Cytoscape,
 * so the rules that decide what a reader sees (costs, stage edges, the hot
 * path) are unit-tested under Node (tests/graph_view/).
 *
 * Reading the same document the profiler writes to disk means the picture
 * and the data file cannot disagree.
 */
(function (globalRoot) {
  "use strict";

  var graphView = globalRoot.mlxPerfGraphView = globalRoot.mlxPerfGraphView || {};

  graphView.CLASSIFICATION_TO_LABEL = {
    compute: "compute",
    wait_on_other_stream: "stalled on another stream",
    gpu_scheduling: "GPU scheduling",
    wait_on_own_stream: "queued on its own stream",
    wait_on_host: "host issue"
  };
  // [document field, reader's name, report colour token], in the order the
  // timeline draws a node's slices.
  graphView.SLICE_FIELDS = [
    ["wait_on_host_us", "host issue", "--slice-cpu"],
    ["wait_on_own_stream_us", "queued", "--slice-queued"],
    ["wait_on_other_stream_us", "stalled", "--slice-stalled"],
    ["gpu_scheduling_us", "GPU scheduling", "--slice-scheduling"],
    ["compute_us", "compute", "--slice-compute"],
    ["result_latency_us", "result", "--slice-result"]
  ];
  // Long fused names ("CompiledExpBroadcastBroadcastAddAsType…") would make
  // boxes too wide to lay out; the full name is in the tooltip.
  var MAX_DISPLAYED_OP_NAME_LENGTH = 26;

  graphView.formatMicroseconds = function (microseconds) {
    if (typeof microseconds !== "number" || isNaN(microseconds)) { return "n/a"; }
    if (Math.abs(microseconds) < 1000) { return microseconds.toFixed(1) + " µs"; }
    if (Math.abs(microseconds) < 1e6) { return (microseconds / 1000).toFixed(2) + " ms"; }
    return (microseconds / 1e6).toFixed(2) + " s";
  };

  graphView.formatBytes = function (byteCount) {
    var units = ["B", "KiB", "MiB", "GiB"];
    var scaled = typeof byteCount === "number" ? byteCount : 0;
    var unitIndex = 0;
    while (scaled >= 1024 && unitIndex < units.length - 1) { scaled /= 1024; unitIndex += 1; }
    return (unitIndex === 0 ? String(scaled) : scaled.toFixed(1)) + " " + units[unitIndex];
  };

  graphView.shortenOpName = function (opName) {
    return opName.length > MAX_DISPLAYED_OP_NAME_LENGTH
      ? opName.slice(0, MAX_DISPLAYED_OP_NAME_LENGTH - 1) + "…" : opName;
  };

  // A timed unit's cost is its GPU-measured compute. A stage-timed job's
  // nodes were not timed one by one, so their cost is the serial pass's
  // time for the node run alone, and the caption says so.
  graphView.describeNodeCost = function (nodeRecord) {
    if (typeof nodeRecord.compute_us === "number") {
      return { microseconds: nodeRecord.compute_us,
               prefix: nodeRecord.is_compute_gpu_measured ? "gpu " : "~" };
    }
    if (typeof nodeRecord.serial_duration_us === "number") {
      return { microseconds: nodeRecord.serial_duration_us, prefix: "alone " };
    }
    return { microseconds: 0, prefix: "" };
  };

  graphView.buildGraphModel = function (measurementDocument) {
    var nodeRecords = measurementDocument.nodes || [];
    var stageRecords = measurementDocument.stages || [];
    var positionToNode = {};
    var nodeToStage = {};
    nodeRecords.forEach(function (nodeRecord) { positionToNode[nodeRecord.position] = nodeRecord; });
    stageRecords.forEach(function (stageRecord) {
      stageRecord.node_positions.forEach(function (position) { nodeToStage[position] = stageRecord.stage_position; });
    });
    return {
      variantName: measurementDocument.variant_name || "variant",
      isStageTimed: measurementDocument.granularity === "per_stage" && stageRecords.length > 0,
      nodeRecords: nodeRecords,
      stageRecords: stageRecords,
      positionToNode: positionToNode,
      nodeToStage: nodeToStage
    };
  };

  function nodeElement(nodeRecord, maximumCost) {
    var cost = graphView.describeNodeCost(nodeRecord);
    var displayName = graphView.shortenOpName(nodeRecord.kernel_name || nodeRecord.op_name || nodeRecord.label);
    return { group: "nodes", data: {
      id: "n" + nodeRecord.position, kind: "node", order: nodeRecord.position, position: nodeRecord.position,
      caption: displayName + "\n" + cost.prefix + graphView.formatMicroseconds(cost.microseconds) +
               " · " + graphView.formatBytes(nodeRecord.output_bytes),
      cost: cost.microseconds, costShare: maximumCost > 0 ? cost.microseconds / maximumCost : 0,
      classification: nodeRecord.classification, streamIndex: nodeRecord.stream_index,
      searchText: [nodeRecord.op_name, nodeRecord.kernel_name, nodeRecord.label].join(" ").toLowerCase(),
      width: Math.max(104, Math.min(240, 7.2 * Math.max(displayName.length, 17)))
    } };
  }

  function maximumNodeCost(nodeRecords) {
    return nodeRecords.reduce(function (maximum, nodeRecord) {
      return Math.max(maximum, graphView.describeNodeCost(nodeRecord).microseconds);
    }, 0);
  }

  // Every primitive (or only `memberPositions`) and the producer → consumer
  // edges among them. Input leaves (weights, user inputs) are not nodes in
  // the document: they hold data and do no work, and on a model they would
  // bury the graph.
  graphView.buildNodeElements = function (graphModel, memberPositions) {
    var memberSet = memberPositions ? new Set(memberPositions) : null;
    var shownRecords = graphModel.nodeRecords.filter(function (nodeRecord) {
      return !memberSet || memberSet.has(nodeRecord.position);
    });
    var maximumCost = maximumNodeCost(shownRecords);
    var elements = shownRecords.map(function (nodeRecord) { return nodeElement(nodeRecord, maximumCost); });
    shownRecords.forEach(function (nodeRecord) {
      var crossStreamPositions = new Set(nodeRecord.cross_stream_input_positions || []);
      new Set(nodeRecord.input_positions || []).forEach(function (producerPosition) {
        var producerRecord = graphModel.positionToNode[producerPosition];
        if (!producerRecord || (memberSet && !memberSet.has(producerPosition))) { return; }
        elements.push({ group: "edges", data: {
          id: "e" + producerPosition + "_" + nodeRecord.position,
          source: "n" + producerPosition, target: "n" + nodeRecord.position,
          bytes: producerRecord.output_bytes, isCrossStream: crossStreamPositions.has(producerPosition) } });
      });
    });
    return elements;
  };

  function stageElement(stageRecord, maximumCost) {
    var computeMicroseconds = typeof stageRecord.compute_us === "number" ? stageRecord.compute_us : 0;
    var stageName = stageRecord.stage_name || "stage " + stageRecord.stage_position;
    return { group: "nodes", data: {
      id: "s" + stageRecord.stage_position, kind: "stage", order: stageRecord.stage_position,
      stagePosition: stageRecord.stage_position,
      caption: stageName + "\n" + stageRecord.node_positions.length + " nodes · gpu " +
               graphView.formatMicroseconds(computeMicroseconds),
      cost: computeMicroseconds, costShare: maximumCost > 0 ? computeMicroseconds / maximumCost : 0,
      classification: stageRecord.classification, streamIndex: stageRecord.stream_index,
      searchText: stageName.toLowerCase(), width: Math.max(160, 8 * stageName.length)
    } };
  }

  // Calls visitor(producerPosition, consumerPosition, bytes) once per
  // producer → consumer pair, in document order.
  function forEachDataEdge(graphModel, visitor) {
    graphModel.nodeRecords.forEach(function (nodeRecord) {
      new Set(nodeRecord.input_positions || []).forEach(function (producerPosition) {
        var producerRecord = graphModel.positionToNode[producerPosition];
        if (producerRecord) { visitor(producerPosition, nodeRecord.position, producerRecord.output_bytes); }
      });
    });
  }

  // One block per stage, and one edge per pair of stages that pass data,
  // carrying the bytes summed over every producer → consumer pair.
  graphView.buildStageElements = function (graphModel) {
    var maximumCost = graphModel.stageRecords.reduce(function (maximum, stageRecord) {
      return Math.max(maximum, typeof stageRecord.compute_us === "number" ? stageRecord.compute_us : 0);
    }, 0);
    var elements = graphModel.stageRecords.map(function (stageRecord) { return stageElement(stageRecord, maximumCost); });
    var stagePairToBytes = new Map();
    forEachDataEdge(graphModel, function (producerPosition, consumerPosition, byteCount) {
      var producerStage = graphModel.nodeToStage[producerPosition];
      var consumerStage = graphModel.nodeToStage[consumerPosition];
      if (producerStage === undefined || consumerStage === undefined || producerStage === consumerStage) { return; }
      var pairKey = producerStage + "_" + consumerStage;
      stagePairToBytes.set(pairKey, (stagePairToBytes.get(pairKey) || 0) + byteCount);
    });
    stagePairToBytes.forEach(function (byteCount, pairKey) {
      var stagePositions = pairKey.split("_");
      elements.push({ group: "edges", data: { id: "se" + pairKey, source: "s" + stagePositions[0],
        target: "s" + stagePositions[1], bytes: byteCount, isCrossStream: false } });
    });
    return elements;
  };

  function ghostElement(graphModel, neighbourStage, isUpstream) {
    var stageRecord = graphModel.stageRecords[neighbourStage];
    return { group: "nodes", data: {
      id: (isUpstream ? "from" : "to") + neighbourStage, kind: "ghost", stagePosition: neighbourStage,
      order: isUpstream ? -1 : Number.MAX_SAFE_INTEGER,
      caption: (isUpstream ? "from " : "to ") + (stageRecord.stage_name || "stage " + neighbourStage),
      cost: 0, costShare: 0, searchText: "", width: 128 } };
  }

  // A stage's own nodes, plus one ghost per neighbouring stage it reads from
  // or feeds, so the drill-down shows where data enters and leaves.
  graphView.buildStageDrillDownElements = function (graphModel, stagePosition) {
    var memberPositions = graphModel.stageRecords[stagePosition].node_positions;
    var elements = graphView.buildNodeElements(graphModel, memberPositions);
    var memberSet = new Set(memberPositions);
    var addedIds = new Set();
    function addOnce(element) {
      if (!addedIds.has(element.data.id)) { addedIds.add(element.data.id); elements.push(element); }
    }
    forEachDataEdge(graphModel, function (producerPosition, consumerPosition) {
      var isConsumerInside = memberSet.has(consumerPosition);
      var neighbourStage = graphModel.nodeToStage[isConsumerInside ? producerPosition : consumerPosition];
      if (memberSet.has(producerPosition) === isConsumerInside || neighbourStage === undefined) { return; }
      var ghost = ghostElement(graphModel, neighbourStage, isConsumerInside);
      addOnce(ghost);
      var sourceId = isConsumerInside ? ghost.data.id : "n" + producerPosition;
      var targetId = isConsumerInside ? "n" + consumerPosition : ghost.data.id;
      addOnce({ group: "edges", data: { id: "g" + sourceId + "_" + targetId, source: sourceId,
        target: targetId, bytes: 0, isCrossStream: false, isGhostEdge: true } });
    });
    return elements;
  };

  // True when the edges join each stage only to the next one in dispatch
  // order (a decoder's layers): a layered layout would then draw one tall
  // column, too small to read once it is fitted to the screen.
  graphView.isSimpleStageChain = function (stageElements) {
    var stageCount = stageElements.filter(function (element) { return element.group === "nodes"; }).length;
    var edges = stageElements.filter(function (element) { return element.group === "edges"; });
    return stageCount > 1 && edges.length === stageCount - 1 && edges.every(function (edge) {
      return Number(edge.data.target.slice(1)) === Number(edge.data.source.slice(1)) + 1;
    });
  };

  // Positions for a chain laid out as a snake: left to right along the
  // first row, right to left along the next, so every edge joins two
  // neighbouring tiles and the whole chain fits a wide screen.
  graphView.snakePositions = function (tileCount, columnCount, columnPitch, rowPitch) {
    var columns = Math.max(1, Math.min(columnCount, tileCount));
    var positions = [];
    for (var tileIndex = 0; tileIndex < tileCount; tileIndex += 1) {
      var rowIndex = Math.floor(tileIndex / columns);
      var placeInRow = tileIndex % columns;
      var columnIndex = rowIndex % 2 === 0 ? placeInRow : columns - 1 - placeInRow;
      positions.push({ x: columnIndex * columnPitch, y: rowIndex * rowPitch });
    }
    return positions;
  };

  // The most expensive dependency chain: the longest path by cost through a
  // DAG. `orderedNodes` must be topologically ordered ({id, cost}), which
  // document positions are, so one pass suffices.
  graphView.findMostExpensiveChain = function (orderedNodes, nodeIdToProducerIds) {
    var nodeIdToBest = new Map();
    var bestEnd = null;
    orderedNodes.forEach(function (graphNode) {
      var bestIncoming = null;
      (nodeIdToProducerIds[graphNode.id] || []).forEach(function (producerId) {
        var producerBest = nodeIdToBest.get(producerId);
        if (producerBest && (!bestIncoming || producerBest.total > bestIncoming.total)) { bestIncoming = producerBest; }
      });
      var chainEnd = { total: (bestIncoming ? bestIncoming.total : 0) + graphNode.cost,
                       previous: bestIncoming, id: graphNode.id };
      nodeIdToBest.set(graphNode.id, chainEnd);
      if (!bestEnd || chainEnd.total > bestEnd.total) { bestEnd = chainEnd; }
    });
    var chainIds = [];
    for (var chainLink = bestEnd; chainLink; chainLink = chainLink.previous) { chainIds.unshift(chainLink.id); }
    return { nodeIds: chainIds, totalMicroseconds: bestEnd ? bestEnd.total : 0 };
  };
})(typeof window !== "undefined" ? window : globalThis);
