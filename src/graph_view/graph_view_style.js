/*
 * mlx-performance-profiler graph view: colours and the Cytoscape style.
 *
 * Colours come from the report's own CSS tokens (--slice-compute, …), read
 * at mount time, so a node is the same colour as its bar in the timeline
 * and the graph follows the page between light and dark themes.
 */
(function (globalRoot) {
  "use strict";

  var graphView = globalRoot.mlxPerfGraphView = globalRoot.mlxPerfGraphView || {};

  var CLASSIFICATION_TO_TOKEN = {
    compute: "--slice-compute",
    wait_on_other_stream: "--slice-stalled",
    gpu_scheduling: "--slice-scheduling",
    wait_on_own_stream: "--slice-queued",
    wait_on_host: "--slice-cpu"
  };
  // Cool to hot, an inferno-like ramp that shares no hue with the slice
  // colours, so "cost" is never mistaken for "compute" or "stalled". Cost
  // shares are square-rooted first so one dominant node does not wash every
  // other node out to the coolest colour.
  graphView.HEAT_COLOURS = ["#4b3f8f", "#8e3f9e", "#d0457a", "#f2804a", "#fcd25a"];
  graphView.STREAM_COLOURS = ["#6b9bf2", "#3fb77a", "#e8a43a", "#b07ce8", "#3cc4c4", "#f07ab0"];
  var HOT_PATH_COLOUR = "#ffb224";
  var FONT_FAMILY = "-apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif";

  function mixColours(fromHex, toHex, fraction) {
    var fromValue = parseInt(fromHex.slice(1), 16);
    var toValue = parseInt(toHex.slice(1), 16);
    return "#" + [16, 8, 0].map(function (shift) {
      var fromChannel = (fromValue >> shift) & 255;
      var toChannel = (toValue >> shift) & 255;
      var mixedChannel = Math.round(fromChannel + (toChannel - fromChannel) * fraction);
      return ("0" + mixedChannel.toString(16)).slice(-2);
    }).join("");
  }

  graphView.heatColour = function (costShare) {
    var heatColours = graphView.HEAT_COLOURS;
    var scaledShare = Math.sqrt(Math.max(0, Math.min(1, costShare || 0)));
    var bandPosition = scaledShare * (heatColours.length - 1);
    var lowerIndex = Math.min(heatColours.length - 2, Math.floor(bandPosition));
    return mixColours(heatColours[lowerIndex], heatColours[lowerIndex + 1], bandPosition - lowerIndex);
  };

  graphView.readTheme = function (themeElement) {
    var computedStyle = getComputedStyle(themeElement);
    function token(name, fallback) { return computedStyle.getPropertyValue(name).trim() || fallback; }
    var theme = { page: token("--page", "#11161d"), ink: token("--ink", "#e4e9f0"),
                  muted: token("--muted", "#94a3b5"), rule: token("--rule", "#2a3441") };
    Object.keys(CLASSIFICATION_TO_TOKEN).forEach(function (classification) {
      theme[classification] = token(CLASSIFICATION_TO_TOKEN[classification], "#888888");
    });
    graphView.SLICE_FIELDS.forEach(function (sliceField) { theme[sliceField[2]] = token(sliceField[2], "#888888"); });
    return theme;
  };

  // colourMode: "time" (where the time went), "heat" (cost) or "stream".
  // Nodes without a classification (a stage-timed job's nodes) fall back to
  // heat, since they have a cost but were not classified one by one.
  graphView.nodeColour = function (theme, colourMode, elementData) {
    if (elementData.kind === "ghost") { return theme.muted; }
    if (colourMode === "stream") {
      return graphView.STREAM_COLOURS[(elementData.streamIndex || 0) % graphView.STREAM_COLOURS.length];
    }
    if (colourMode === "time" && elementData.classification) {
      return theme[elementData.classification] || theme.muted;
    }
    return graphView.heatColour(elementData.costShare);
  };

  function nodeStyles(theme, colourOf) {
    return [
      { selector: "node", style: {
        "shape": "round-rectangle", "width": "data(width)", "height": 42,
        // Cost as fill strength: the expensive nodes are vivid and the
        // cheap ones faint, so the eye finds where the time went first.
        // (Cytoscape's underlay has no blur, so a "glow" drew as a box.)
        "background-color": colourOf,
        "background-opacity": function (element) { return 0.1 + 0.62 * Math.sqrt(element.data("costShare")); },
        "border-width": function (element) { return 1.2 + 1.8 * element.data("costShare"); },
        "border-color": colourOf,
        "label": "data(caption)", "text-wrap": "wrap", "text-valign": "center", "text-halign": "center",
        "color": theme.ink, "font-size": 11, "font-weight": 600, "line-height": 1.35,
        "font-family": FONT_FAMILY, "min-zoomed-font-size": 6,
        "text-outline-color": theme.page, "text-outline-width": 0.6, "text-outline-opacity": 0.55,
        "transition-property": "opacity, border-width", "transition-duration": "160ms" } },
      { selector: "node[kind = 'stage']", style: { "height": 50, "font-size": 12.5 } },
      { selector: "node[kind = 'ghost']", style: { "border-style": "dashed", "background-opacity": 0.04,
        "color": theme.muted, "font-weight": 500, "height": 28 } },
      { selector: "node.focus", style: { "border-width": 3.5 } },
      { selector: "node.match", style: { "border-width": 3.5, "border-color": theme.ink } },
      { selector: "node.hot", style: { "border-width": 3, "border-color": HOT_PATH_COLOUR } }
    ];
  }

  function edgeStyles(theme) {
    return [
      { selector: "edge", style: {
        "curve-style": "bezier", "target-arrow-shape": "triangle", "arrow-scale": 0.8,
        "line-color": theme.muted, "target-arrow-color": theme.muted, "line-opacity": 0.45,
        // Width grows with the bytes the consumer reads, on a log scale:
        // tensors span bytes to gigabytes.
        "width": function (element) { return 1 + Math.min(4, Math.log10(1 + element.data("bytes")) / 2); } } },
      { selector: "edge[?isCrossStream]", style: { "line-color": theme.wait_on_other_stream,
        "target-arrow-color": theme.wait_on_other_stream, "line-style": "dashed", "line-opacity": 0.85 } },
      { selector: "edge[?isGhostEdge]", style: { "line-style": "dotted", "line-opacity": 0.3 } },
      { selector: "edge.hot", style: { "line-color": HOT_PATH_COLOUR, "target-arrow-color": HOT_PATH_COLOUR,
        "line-opacity": 1, "width": 4 } },
      { selector: ".faded", style: { "opacity": 0.12 } }
    ];
  }

  graphView.buildStylesheet = function (theme, colourMode) {
    function colourOf(element) { return graphView.nodeColour(theme, colourMode, element.data()); }
    return nodeStyles(theme, colourOf).concat(edgeStyles(theme));
  };
})(typeof window !== "undefined" ? window : globalThis);
