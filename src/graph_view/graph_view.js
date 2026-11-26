/*
 * mlx-performance-profiler graph view: mounting and interaction.
 *
 * Every element with class "graph-view" whose data-graph-document names a
 * <script type="application/json"> holding a measurement document becomes
 * an explorable graph: hover for a node's six-slice breakdown, click to
 * light up everything upstream and downstream, search, the hot path, and
 * drill-down from stages into their nodes.
 *
 * Rendering is Cytoscape.js on a canvas (a thousand nodes still pan and zoom
 * smoothly) with the dagre layered layout; both are embedded before this
 * script, after graph_view_model.js, graph_view_style.js and
 * graph_view_panels.js.
 */
(function (globalRoot) {
  "use strict";

  var graphView = globalRoot.mlxPerfGraphView = globalRoot.mlxPerfGraphView || {};

  // Above this many nodes the layout takes long enough to warrant a notice,
  // and edges hide while panning so the canvas stays responsive.
  var LARGE_GRAPH_NODE_COUNT = 300;
  var LARGE_LAYOUT_DEFER_MILLISECONDS = 30;
  var TOOLTIP_WIDTH_PIXELS = 300;
  // At 11 px captions, below this zoom the text stops being readable.
  var MINIMUM_READABLE_ZOOM = 0.62;
  var MAXIMUM_FIT_ZOOM = 1.3;

  var VIEW_TEMPLATE =
    "<div class=\"gv-toolbar\"><div class=\"gv-breadcrumb\"></div><div class=\"gv-controls\">" +
    "<span class=\"gv-segmented\" data-role=\"overview-mode\" hidden>" +
    "<button type=\"button\" data-mode=\"stages\" aria-pressed=\"true\">Stages</button>" +
    "<button type=\"button\" data-mode=\"nodes\" aria-pressed=\"false\">All nodes</button></span>" +
    "<input type=\"search\" class=\"gv-search\" placeholder=\"Search ops  /\" aria-label=\"Search nodes by op, kernel or label\">" +
    "<select class=\"gv-colour-mode\" aria-label=\"Colour nodes by\">" +
    "<option value=\"time\">Colour: where time went</option><option value=\"heat\">Colour: cost</option>" +
    "<option value=\"stream\">Colour: stream</option></select>" +
    "<button type=\"button\" data-action=\"hot-path\" aria-pressed=\"false\" title=\"Highlight the most expensive dependency chain\">Hot path</button>" +
    "<button type=\"button\" data-action=\"fit\" title=\"Fit the graph (f)\">Fit</button>" +
    "<button type=\"button\" data-action=\"fullscreen\" title=\"Full screen\">⤢</button>" +
    "<button type=\"button\" data-action=\"png\" title=\"Download the whole graph as PNG\">PNG</button>" +
    "</div></div><div class=\"gv-stage\"><div class=\"gv-canvas\" role=\"img\"></div>" +
    "<div class=\"gv-tooltip\" hidden></div><aside class=\"gv-details\" hidden></aside></div>" +
    "<div class=\"gv-footer\"><div class=\"gv-legend\"></div><div class=\"gv-status\" aria-live=\"polite\"></div></div>";

  var READABLE_ZOOM_HINT = "Drag to pan · scroll to zoom · Fit shows it all";

  function setStatus(viewState, statusText) { viewState.statusLine.textContent = statusText; }

  // Shows the whole graph; the "drag to pan" hint no longer applies after.
  function fitAll(viewState) {
    viewState.cy.fit(undefined, 28);
    if (viewState.statusLine.textContent === READABLE_ZOOM_HINT) { setStatus(viewState, ""); }
  }

  function clearHighlights(viewState) {
    viewState.cy.elements().removeClass("faded focus match");
    viewState.detailsPanel.hidden = true;
  }

  function clearHotPath(viewState) {
    viewState.cy.elements().removeClass("hot");
    viewState.hotPathButton.setAttribute("aria-pressed", "false");
  }

  function focusOnNode(viewState, cyNode) {
    clearHighlights(viewState);
    var neighbourhood = cyNode.predecessors().union(cyNode.successors()).union(cyNode);
    viewState.cy.elements().difference(neighbourhood).addClass("faded");
    cyNode.addClass("focus");
    viewState.detailsPanel.innerHTML = graphView.describeElementHtml(viewState.graphModel, viewState.theme, cyNode.data());
    viewState.detailsPanel.hidden = false;
  }

  function showTooltip(viewState, cyNode) {
    var tooltip = viewState.tooltip;
    tooltip.innerHTML = graphView.describeElementHtml(viewState.graphModel, viewState.theme, cyNode.data());
    var nodeCentre = cyNode.renderedPosition();
    var halfWidth = cyNode.renderedWidth() / 2;
    var isRightHalf = nodeCentre.x > viewState.canvasElement.clientWidth / 2;
    tooltip.style.left = (isRightHalf ? nodeCentre.x - halfWidth - TOOLTIP_WIDTH_PIXELS - 12
                                      : nodeCentre.x + halfWidth + 12) + "px";
    tooltip.style.top = Math.max(6, nodeCentre.y - 40) + "px";
    tooltip.hidden = false;
  }

  function toggleHotPath(viewState) {
    var cy = viewState.cy;
    if (cy.elements(".hot").length > 0) { clearHotPath(viewState); setStatus(viewState, ""); return; }
    var costNodes = cy.nodes("[kind != 'ghost']").sort(function (first, second) {
      return first.data("order") - second.data("order");
    });
    var nodeIdToProducerIds = {};
    costNodes.forEach(function (cyNode) {
      nodeIdToProducerIds[cyNode.id()] = cyNode.incomers("node").map(function (producer) { return producer.id(); });
    });
    var chain = graphView.findMostExpensiveChain(costNodes.map(function (cyNode) {
      return { id: cyNode.id(), cost: cyNode.data("cost") };
    }), nodeIdToProducerIds);
    chain.nodeIds.forEach(function (nodeId, chainIndex) {
      var cyNode = cy.getElementById(nodeId).addClass("hot");
      if (chainIndex > 0) { cy.getElementById(chain.nodeIds[chainIndex - 1]).edgesTo(cyNode).addClass("hot"); }
    });
    viewState.hotPathButton.setAttribute("aria-pressed", "true");
    setStatus(viewState, "Hot path: " + chain.nodeIds.length + (viewState.isShowingStages ? " stages, " : " nodes, ") +
              graphView.formatMicroseconds(chain.totalMicroseconds) + " of cost along one dependency chain");
  }

  function applySearch(viewState, queryText) {
    var cy = viewState.cy;
    cy.elements().removeClass("faded match focus");
    viewState.searchMatches = null;
    var query = queryText.trim().toLowerCase();
    if (!query) { setStatus(viewState, ""); return; }
    var matches = cy.nodes().filter(function (cyNode) { return (cyNode.data("searchText") || "").indexOf(query) >= 0; });
    cy.elements().difference(matches).addClass("faded");
    matches.addClass("match");
    viewState.searchMatches = matches;
    viewState.searchMatchIndex = -1;
    setStatus(viewState, matches.length + (matches.length === 1 ? " match" : " matches") +
              (matches.length > 0 ? " · Enter steps through them" : ""));
  }

  function stepToNextMatch(viewState) {
    var matches = viewState.searchMatches;
    if (!matches || matches.length === 0) { return; }
    viewState.searchMatchIndex = (viewState.searchMatchIndex + 1) % matches.length;
    viewState.cy.animate({ center: { eles: matches[viewState.searchMatchIndex] }, zoom: 1.3, duration: 280 });
  }

  function downloadPng(viewState) {
    var pngBlob = viewState.cy.png({ output: "blob", full: true, scale: 2, bg: viewState.theme.page });
    var downloadLink = document.createElement("a");
    downloadLink.href = URL.createObjectURL(pngBlob);
    downloadLink.download = viewState.graphModel.variantName.replace(/[^A-Za-z0-9_.-]+/g, "_") + "-graph.png";
    document.body.appendChild(downloadLink);
    downloadLink.click();
    downloadLink.remove();
    setTimeout(function () { URL.revokeObjectURL(downloadLink.href); }, 1000);
  }

  function renderLegend(viewState) {
    var shownNodes = viewState.cy.nodes("[kind != 'ghost']");
    var legendFacts = {
      // A filter function, not "[classification]": Cytoscape counts null as
      // defined, and a stage-timed job's nodes carry a null classification.
      shownClassifications: shownNodes.map(function (cyNode) { return cyNode.data("classification"); })
        .filter(function (classification) { return !!classification; }),
      streamIndices: Array.from(new Set(shownNodes.map(function (cyNode) { return cyNode.data("streamIndex") || 0; })))
        .sort(function (first, second) { return first - second; }),
      isCostRunAlone: viewState.graphModel.isStageTimed && !viewState.isShowingStages
    };
    viewState.legend.innerHTML = graphView.renderLegendHtml(viewState.theme, viewState.colourMode, legendFacts);
  }

  var LAYERED_LAYOUT = { name: "dagre", rankDir: "TB", nodeSep: 22, rankSep: 46, edgeSep: 10,
                         animate: false, fit: true, padding: 28 };
  var SNAKE_COLUMN_GAP_PIXELS = 64;
  var SNAKE_ROW_PITCH_PIXELS = 116;

  // A chain of stages as a snake that fills the canvas width; every tile
  // shares the widest tile's width so the columns line up.
  function snakeLayoutFor(viewState, stageElements) {
    var tileElements = stageElements.filter(function (element) { return element.group === "nodes"; });
    var tileWidth = tileElements.reduce(function (widest, element) { return Math.max(widest, element.data.width); }, 0);
    tileElements.forEach(function (element) { element.data.width = tileWidth; });
    var columnPitch = tileWidth + SNAKE_COLUMN_GAP_PIXELS;
    var columnCount = Math.floor((viewState.canvasElement.clientWidth - 56 + SNAKE_COLUMN_GAP_PIXELS) / columnPitch);
    var positions = graphView.snakePositions(tileElements.length, columnCount, columnPitch, SNAKE_ROW_PITCH_PIXELS);
    return { name: "preset", fit: true, padding: 28,
             positions: function (cyNode) { return positions[cyNode.data("order")]; } };
  }

  // A deep graph fitted whole would shrink its text below reading size. A
  // detail view instead opens at a readable zoom on the graph's first rows,
  // like the top of a page; the whole graph is one "Fit" away. Returns
  // whether it zoomed.
  function zoomToReadableTop(cy) {
    // A handful of nodes fitted to the canvas would be blown up to poster
    // size; keep them at a natural size, centred.
    if (cy.zoom() > MAXIMUM_FIT_ZOOM) { cy.zoom(MAXIMUM_FIT_ZOOM); cy.center(); return false; }
    if (cy.zoom() >= MINIMUM_READABLE_ZOOM) { return false; }
    var boundingBox = cy.elements().boundingBox();
    cy.zoom(MINIMUM_READABLE_ZOOM);
    cy.pan({ x: cy.width() / 2 - MINIMUM_READABLE_ZOOM * (boundingBox.x1 + boundingBox.w / 2),
             y: 28 - MINIMUM_READABLE_ZOOM * boundingBox.y1 });
    return true;
  }

  function showElements(viewState, elements, breadcrumbHtml, layoutOptions) {
    var cy = viewState.cy;
    var nodeCount = elements.filter(function (element) { return element.group === "nodes"; }).length;
    viewState.tooltip.hidden = true;
    clearHighlights(viewState);
    clearHotPath(viewState);
    viewState.breadcrumb.innerHTML = breadcrumbHtml;
    viewState.canvasElement.setAttribute("aria-label", viewState.graphModel.variantName + " graph: " + nodeCount + " nodes");
    var isLargeGraph = nodeCount > LARGE_GRAPH_NODE_COUNT;
    function layOut() {
      cy.batch(function () { cy.elements().remove(); cy.add(elements); });
      cy.layout(layoutOptions || LAYERED_LAYOUT).run();
      var isZoomedToReadable = !isLargeGraph && zoomToReadableTop(cy);
      setStatus(viewState, nodeCount === 0 ? "No nodes were executed."
        : isZoomedToReadable ? READABLE_ZOOM_HINT : "");
      renderLegend(viewState);
    }
    if (!isLargeGraph) { layOut(); return; }
    // A large layout blocks for a second or more: show the notice first, and
    // lay out on a later task so the browser can paint it. Small graphs lay
    // out at once, so whatever a reader or script does next sees the new view.
    setStatus(viewState, "Laying out " + nodeCount.toLocaleString() + " nodes…");
    setTimeout(layOut, LARGE_LAYOUT_DEFER_MILLISECONDS);
  }

  function showOverview(viewState) {
    var graphModel = viewState.graphModel;
    viewState.isShowingStages = graphModel.isStageTimed && viewState.overviewMode === "stages";
    if (viewState.isShowingStages) {
      var stageElements = graphView.buildStageElements(graphModel);
      showElements(viewState, stageElements,
                   "<b>" + graphModel.stageRecords.length + " stages</b><span> · click a stage to open it</span>",
                   graphView.isSimpleStageChain(stageElements) ? snakeLayoutFor(viewState, stageElements) : null);
    } else {
      showElements(viewState, graphView.buildNodeElements(graphModel, null),
                   "<b>" + graphModel.nodeRecords.length.toLocaleString() + " nodes</b><span> · click a node to trace it</span>");
    }
  }

  function openStage(viewState, stagePosition) {
    var stageRecord = viewState.graphModel.stageRecords[stagePosition];
    viewState.isShowingStages = false;
    showElements(viewState, graphView.buildStageDrillDownElements(viewState.graphModel, stagePosition),
                 "<a href=\"#\" data-action=\"overview\">All stages</a> › <b>" +
                 graphView.escapeHtml(stageRecord.stage_name) + "</b><span> · " +
                 stageRecord.node_positions.length + " nodes</span>");
  }

  function restyle(viewState) {
    viewState.theme = graphView.readTheme(viewState.rootElement);
    viewState.cy.style(graphView.buildStylesheet(viewState.theme, viewState.colourMode));
    renderLegend(viewState);
  }

  function setOverviewMode(viewState, overviewMode) {
    viewState.overviewMode = overviewMode;
    viewState.rootElement.querySelectorAll("[data-role='overview-mode'] button").forEach(function (modeButton) {
      modeButton.setAttribute("aria-pressed", String(modeButton.getAttribute("data-mode") === overviewMode));
    });
    showOverview(viewState);
  }

  function toggleFullscreen(viewState) {
    if (document.fullscreenElement === viewState.rootElement) { document.exitFullscreen(); }
    else if (viewState.rootElement.requestFullscreen) { viewState.rootElement.requestFullscreen(); }
  }

  function onCanvasTap(viewState, tapEvent) {
    var tapped = tapEvent.target;
    if (tapped === viewState.cy) { clearHighlights(viewState); return; }
    if (!tapped.isNode || !tapped.isNode()) { return; }
    var kind = tapped.data("kind");
    if (kind === "stage" || kind === "ghost") { openStage(viewState, tapped.data("stagePosition")); }
    else { focusOnNode(viewState, tapped); }
  }

  function bindCanvasEvents(viewState) {
    var cy = viewState.cy;
    cy.on("tap", function (tapEvent) { onCanvasTap(viewState, tapEvent); });
    cy.on("mouseover", "node", function (hoverEvent) { showTooltip(viewState, hoverEvent.target); });
    cy.on("mouseout", "node", function () { viewState.tooltip.hidden = true; });
    cy.on("viewport", function () { viewState.tooltip.hidden = true; });
  }

  function onToolbarClick(viewState, clickEvent) {
    var actionElement = clickEvent.target.closest("[data-action], [data-mode]");
    if (!actionElement) { return; }
    clickEvent.preventDefault();
    var action = actionElement.getAttribute("data-action");
    if (actionElement.hasAttribute("data-mode")) { setOverviewMode(viewState, actionElement.getAttribute("data-mode")); }
    else if (action === "overview") { showOverview(viewState); }
    else if (action === "hot-path") { toggleHotPath(viewState); }
    else if (action === "fit") { fitAll(viewState); }
    else if (action === "fullscreen") { toggleFullscreen(viewState); }
    else if (action === "png") { downloadPng(viewState); }
  }

  function onKeyDown(viewState, keyEvent) {
    var isTyping = keyEvent.target === viewState.searchInput;
    if (keyEvent.key === "Enter" && isTyping) { stepToNextMatch(viewState); return; }
    if (keyEvent.key === "Escape") {
      viewState.searchInput.value = ""; applySearch(viewState, ""); clearHighlights(viewState); return;
    }
    if (isTyping) { return; }
    if (keyEvent.key === "/") { keyEvent.preventDefault(); viewState.searchInput.focus(); }
    else if (keyEvent.key === "f") { fitAll(viewState); }
  }

  function bindControls(viewState) {
    var rootElement = viewState.rootElement;
    rootElement.addEventListener("click", function (clickEvent) { onToolbarClick(viewState, clickEvent); });
    rootElement.addEventListener("keydown", function (keyEvent) { onKeyDown(viewState, keyEvent); });
    viewState.searchInput.addEventListener("input", function () { applySearch(viewState, viewState.searchInput.value); });
    rootElement.querySelector(".gv-colour-mode").addEventListener("change", function (changeEvent) {
      viewState.colourMode = changeEvent.target.value;
      restyle(viewState);
    });
    document.addEventListener("fullscreenchange", function () { viewState.cy.resize(); fitAll(viewState); });
    if (globalRoot.matchMedia) {
      globalRoot.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () { restyle(viewState); });
    }
  }

  function createViewState(rootElement, measurementDocument) {
    rootElement.innerHTML = VIEW_TEMPLATE;
    var graphModel = graphView.buildGraphModel(measurementDocument);
    var viewState = {
      rootElement: rootElement, graphModel: graphModel, colourMode: "time", overviewMode: "stages",
      isShowingStages: false, searchMatches: null, searchMatchIndex: -1,
      theme: graphView.readTheme(rootElement),
      breadcrumb: rootElement.querySelector(".gv-breadcrumb"),
      canvasElement: rootElement.querySelector(".gv-canvas"),
      tooltip: rootElement.querySelector(".gv-tooltip"),
      detailsPanel: rootElement.querySelector(".gv-details"),
      legend: rootElement.querySelector(".gv-legend"),
      statusLine: rootElement.querySelector(".gv-status"),
      searchInput: rootElement.querySelector(".gv-search"),
      hotPathButton: rootElement.querySelector("[data-action='hot-path']")
    };
    rootElement.querySelector("[data-role='overview-mode']").hidden = !graphModel.isStageTimed;
    viewState.cy = globalRoot.cytoscape({
      container: viewState.canvasElement, style: graphView.buildStylesheet(viewState.theme, viewState.colourMode),
      minZoom: 0.04, maxZoom: 3, boxSelectionEnabled: false, autoungrabify: true,
      hideEdgesOnViewport: graphModel.nodeRecords.length > LARGE_GRAPH_NODE_COUNT
    });
    return viewState;
  }

  graphView.mount = function (rootElement) {
    var documentElement = document.getElementById(rootElement.getAttribute("data-graph-document"));
    if (!documentElement || !globalRoot.cytoscape) {
      rootElement.textContent = "The graph view could not load its data.";
      return null;
    }
    var measurementDocument;
    try {
      measurementDocument = JSON.parse(documentElement.textContent);
    } catch (parseError) {
      // One unreadable document must not stop the page's other views.
      rootElement.textContent = "The graph view could not read its measurement document.";
      return null;
    }
    var viewState = createViewState(rootElement, measurementDocument);
    bindCanvasEvents(viewState);
    bindControls(viewState);
    showOverview(viewState);
    return createController(viewState);
  };

  // What a page script (or a screenshot tool) may do to a mounted view, the
  // same things a reader can do with the toolbar. A view of up to
  // LARGE_GRAPH_NODE_COUNT nodes is laid out before each call returns; a
  // larger one a moment later.
  function createController(viewState) {
    return {
      cy: viewState.cy,
      openStage: function (stagePosition) { openStage(viewState, stagePosition); },
      showOverview: function (overviewMode) { setOverviewMode(viewState, overviewMode || "stages"); },
      setColourMode: function (colourMode) {
        viewState.colourMode = colourMode;
        viewState.rootElement.querySelector(".gv-colour-mode").value = colourMode;
        restyle(viewState);
      },
      showHotPath: function () { if (viewState.cy.elements(".hot").length === 0) { toggleHotPath(viewState); } },
      fit: function () { fitAll(viewState); }
    };
  }

  graphView.mountAll = function () {
    if (globalRoot.cytoscape && globalRoot.cytoscapeDagre && !graphView.isDagreRegistered) {
      globalRoot.cytoscape.use(globalRoot.cytoscapeDagre);
      graphView.isDagreRegistered = true;
    }
    // Kept on the namespace so a page script (or a screenshot tool) can
    // drive a view: open a stage, switch colours, fit.
    graphView.mountedViews = Array.prototype.map.call(
      document.querySelectorAll(".graph-view[data-graph-document]"), graphView.mount);
    return graphView.mountedViews;
  };

  if (typeof document !== "undefined") {
    if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", graphView.mountAll); }
    else { graphView.mountAll(); }
  }
})(typeof window !== "undefined" ? window : globalThis);
