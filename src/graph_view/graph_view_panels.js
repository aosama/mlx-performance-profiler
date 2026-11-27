/*
 * mlx-performance-profiler graph view: the HTML around the canvas — the
 * hover card, the details panel and the legend. Every value from the
 * document is escaped: op and kernel names are author-chosen text.
 */
(function (globalRoot) {
  "use strict";

  var graphView = globalRoot.mlxPerfGraphView = globalRoot.mlxPerfGraphView || {};

  graphView.escapeHtml = function (text) {
    return String(text).replace(/[&<>"']/g, function (character) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[character];
    });
  };
  var escapeHtml = graphView.escapeHtml;
  var formatMicroseconds = graphView.formatMicroseconds;

  // The six slices of one timed unit as a stacked bar plus the non-zero
  // rows, matching the timeline's colours.
  function renderSliceBreakdown(timedRecord, theme) {
    var wall = timedRecord.wall_us;
    if (typeof wall !== "number" || wall <= 0) { return ""; }
    var sliceFields = graphView.SLICE_FIELDS;
    var segments = sliceFields.map(function (sliceField) {
      var sliceMicroseconds = timedRecord[sliceField[0]] || 0;
      return "<span style=\"width:" + (100 * sliceMicroseconds / wall).toFixed(2) + "%;background:" +
             theme[sliceField[2]] + "\"></span>";
    }).join("");
    var rows = sliceFields.filter(function (sliceField) { return (timedRecord[sliceField[0]] || 0) > 0; })
      .map(function (sliceField) {
        return "<tr><td><i style=\"background:" + theme[sliceField[2]] + "\"></i>" + sliceField[1] +
               "</td><td>" + formatMicroseconds(timedRecord[sliceField[0]]) + "</td></tr>";
      }).join("");
    return "<div class=\"gv-slice-bar\">" + segments + "</div><table class=\"gv-facts\">" + rows +
           "<tr class=\"gv-total\"><td>wall</td><td>" + formatMicroseconds(wall) + "</td></tr></table>";
  }

  function renderFacts(facts) {
    return "<table class=\"gv-facts\">" + facts.map(function (fact) {
      return "<tr><td>" + fact[0] + "</td><td>" + escapeHtml(fact[1]) + "</td></tr>";
    }).join("") + "</table>";
  }

  function renderClassification(classification, suffix) {
    return classification ? "<p class=\"gv-class\">mostly " + graphView.CLASSIFICATION_TO_LABEL[classification] +
                            (suffix || "") + "</p>" : "";
  }

  // Each kernel launch: its Metal function, threadgroups, and how many GPU
  // cores it can keep busy. Long fused-kernel names are cut; the full name
  // is the row's tooltip.
  var MAXIMUM_LAUNCHES_SHOWN = 6;
  function renderLaunches(nodeRecord, gpuCoreCount) {
    var launches = nodeRecord.kernel_dispatches;
    if (!launches || launches.length === 0) { return ""; }
    var rows = launches.slice(0, MAXIMUM_LAUNCHES_SHOWN).map(function (launch) {
      var functionName = launch.kernel_function_name || "(unnamed pipeline)";
      var isUnderfilling = gpuCoreCount > 0 && launch.threadgroup_count < gpuCoreCount;
      var coreText = gpuCoreCount > 0
        ? (isUnderfilling ? "⚠ " : "") + Math.min(launch.threadgroup_count, gpuCoreCount) + " of " + gpuCoreCount + " cores"
        : "";
      return "<tr title=\"" + escapeHtml(functionName) + "\"><td>" + escapeHtml(graphView.shortenOpName(functionName)) +
             "<br><small>" + launch.threadgroup_count + " × " + launch.threadgroup_thread_count + " threads</small></td>" +
             "<td" + (isUnderfilling ? " class=\"gv-underfills\"" : "") + ">" + coreText + "</td></tr>";
    }).join("");
    var moreText = launches.length > MAXIMUM_LAUNCHES_SHOWN
      ? "<p class=\"gv-note\">and " + (launches.length - MAXIMUM_LAUNCHES_SHOWN) + " more launches</p>" : "";
    return "<p class=\"gv-section-title\">Kernel launches</p><table class=\"gv-facts gv-launches\">" + rows + "</table>" + moreText;
  }

  function describeNodeHtml(graphModel, nodeRecord, theme) {
    var stagePosition = graphModel.nodeToStage[nodeRecord.position];
    var stageRecord = stagePosition === undefined ? null : graphModel.stageRecords[stagePosition];
    var isTimedByStage = stageRecord !== null && typeof nodeRecord.wall_us !== "number";
    var facts = [["shape", "[" + (nodeRecord.shape || []).join(", ") + "] " + nodeRecord.dtype],
                 ["output", graphView.formatBytes(nodeRecord.output_bytes)],
                 ["stream", nodeRecord.stream_index],
                 ["run alone", formatMicroseconds(nodeRecord.serial_duration_us)]];
    if (nodeRecord.kernel_name) { facts.unshift(["kernel", nodeRecord.kernel_name]); }
    if (stageRecord) { facts.push(["stage", stageRecord.stage_name]); }
    return "<h4>" + escapeHtml(nodeRecord.op_name || nodeRecord.label) + " <small>" +
           escapeHtml(nodeRecord.label) + "</small></h4>" +
           renderClassification(isTimedByStage ? stageRecord.classification : nodeRecord.classification,
                                isTimedByStage ? " (its stage)" : "") +
           renderFacts(facts) + renderSliceBreakdown(isTimedByStage ? stageRecord : nodeRecord, theme) +
           renderLaunches(nodeRecord, graphModel.gpuCoreCount) +
           (isTimedByStage ? "<p class=\"gv-note\">Timed as part of its stage: the bar is the stage's," +
                             " \"run alone\" is this node by itself.</p>" : "");
  }

  function describeStageHtml(stageRecord, theme) {
    return "<h4>" + escapeHtml(stageRecord.stage_name) + "</h4>" +
           renderClassification(stageRecord.classification) +
           renderFacts([["nodes", stageRecord.node_positions.length], ["stream", stageRecord.stream_index]]) +
           renderSliceBreakdown(stageRecord, theme) + "<p class=\"gv-note\">Click to open this stage.</p>";
  }

  graphView.describeElementHtml = function (graphModel, theme, elementData) {
    if (elementData.kind === "stage") {
      return describeStageHtml(graphModel.stageRecords[elementData.stagePosition], theme);
    }
    if (elementData.kind === "ghost") {
      return "<h4>" + escapeHtml(elementData.caption) + "</h4><p class=\"gv-note\">Click to open that stage.</p>";
    }
    return describeNodeHtml(graphModel, graphModel.positionToNode[elementData.position], theme);
  };

  function legendEntries(theme, colourMode, shownClassifications, streamIndices, isCostRunAlone) {
    if (colourMode === "coverage") {
      return [[graphView.coverageColour(theme, 0.25), "few GPU cores busy"],
              [graphView.coverageColour(theme, 1), "every core busy"], [theme.muted, "no kernel launch"]];
    }
    if (colourMode === "stream") {
      return streamIndices.map(function (streamIndex) {
        return [graphView.STREAM_COLOURS[streamIndex % graphView.STREAM_COLOURS.length], "stream " + streamIndex];
      });
    }
    if (colourMode === "time" && shownClassifications.length > 0) {
      return Object.keys(graphView.CLASSIFICATION_TO_LABEL).map(function (classification) {
        return [theme[classification], "mostly " + graphView.CLASSIFICATION_TO_LABEL[classification]];
      });
    }
    var heatColours = graphView.HEAT_COLOURS;
    return [[heatColours[0], "cheapest"], [heatColours[2], "→"],
            [heatColours[4], "most expensive" + (isCostRunAlone ? " (each node run alone)" : "")]];
  }

  graphView.renderLegendHtml = function (theme, colourMode, legendFacts) {
    return legendEntries(theme, colourMode, legendFacts.shownClassifications, legendFacts.streamIndices,
                         legendFacts.isCostRunAlone)
      .map(function (entry) {
        return "<span><i style=\"background:" + entry[0] + "\"></i>" + escapeHtml(entry[1]) + "</span>";
      }).join("") + "<span class=\"gv-legend-note\">brighter = costlier · dashed red = read from another stream" +
      " · ⚠ = a launch leaves GPU cores idle</span>";
  };
})(typeof window !== "undefined" ? window : globalThis);
