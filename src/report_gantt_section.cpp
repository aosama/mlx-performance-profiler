#include "report_gantt_section.h"

#include "latency_slices.h"
#include "report_components.h"
#include "text_format.h"

#include <array>
#include <cmath>
#include <map>
#include <sstream>

namespace mlx_plan {
namespace {

constexpr double LABEL_GUTTER_PIXELS = 140.0;
constexpr double PLOT_WIDTH_PIXELS = 1000.0;
constexpr double AXIS_HEIGHT_PIXELS = 26.0;
constexpr double LANE_HEIGHT_PIXELS = 22.0;
constexpr double LANE_GAP_PIXELS = 8.0;
// Sub-pixel segments still get drawn: a 5 µs kernel in a 3 ms pass must stay
// visible, at the cost of slightly overstating its width.
constexpr double MINIMUM_SEGMENT_PIXELS = 0.8;
constexpr int TARGET_TICK_COUNT = 8;
constexpr int HOST_LANE_INDEX = 0;

// Boundaries of the six causal slices: instant[k] is where slice k starts,
// instant[k+1] where it ends (µs since pass start).
using SliceInstants = std::array<double, CAUSAL_SLICE_ORDER.size() + 1>;

SliceInstants compute_slice_instants(const NodeTimeline& node_timeline) {
  SliceInstants slice_instants{};
  slice_instants[0] = node_timeline.timestamps.issue_started_microseconds;
  for (std::size_t slice_position = 0; slice_position < CAUSAL_SLICE_ORDER.size();
       ++slice_position) {
    slice_instants[slice_position + 1] = slice_instants[slice_position] +
        slice_microseconds(node_timeline.breakdown,
                           CAUSAL_SLICE_ORDER[slice_position]);
  }
  return slice_instants;
}

// Index into SliceInstants where `latency_slice` ends, derived from the
// causal order so reordering slices can never silently misplace an arrow.
std::size_t slice_end_index(LatencySlice latency_slice) {
  for (std::size_t slice_position = 0; slice_position < CAUSAL_SLICE_ORDER.size();
       ++slice_position) {
    if (CAUSAL_SLICE_ORDER[slice_position] == latency_slice) {
      return slice_position + 1;
    }
  }
  return CAUSAL_SLICE_ORDER.size();
}

struct GanttScale {
  double total_microseconds = 1.0;
  double x_for(double instant_microseconds) const {
    return LABEL_GUTTER_PIXELS +
        PLOT_WIDTH_PIXELS * instant_microseconds / total_microseconds;
  }
  static double lane_top(int lane_index) {
    return AXIS_HEIGHT_PIXELS + lane_index * (LANE_HEIGHT_PIXELS + LANE_GAP_PIXELS);
  }
};

// A 1/2/5 × 10^k step gives round tick labels for any pass length.
double choose_tick_step(double total_microseconds) {
  const double raw_step = total_microseconds / TARGET_TICK_COUNT;
  const double magnitude = std::pow(10.0, std::floor(std::log10(raw_step)));
  for (const double multiplier : {1.0, 2.0, 5.0}) {
    if (multiplier * magnitude >= raw_step) {
      return multiplier * magnitude;
    }
  }
  return 10.0 * magnitude;
}

std::string render_axis(const GanttScale& scale, double chart_height) {
  std::ostringstream axis_svg;
  const double tick_step = choose_tick_step(scale.total_microseconds);
  for (double tick = 0.0; tick <= scale.total_microseconds + 1e-9;
       tick += tick_step) {
    const double tick_x = scale.x_for(tick);
    axis_svg << "<line class=\"grid\" x1=\"" << tick_x << "\" y1=\""
             << AXIS_HEIGHT_PIXELS - 4 << "\" x2=\"" << tick_x << "\" y2=\""
             << chart_height << "\"/><text class=\"axis-text\" x=\"" << tick_x
             << "\" y=\"14\" text-anchor=\"middle\">"
             << escape_html_text(format_microseconds(tick)) << "</text>";
  }
  return axis_svg.str();
}

void append_segment(
    std::ostringstream& segments_svg,
    const GanttScale& scale,
    int lane_index,
    double started_microseconds,
    double finished_microseconds,
    LatencySlice latency_slice,
    const std::string& node_caption) {
  const double duration_microseconds =
      finished_microseconds - started_microseconds;
  if (duration_microseconds <= 0.0) {
    return;
  }
  const double segment_width = std::max(
      MINIMUM_SEGMENT_PIXELS,
      scale.x_for(finished_microseconds) - scale.x_for(started_microseconds));
  const SliceStyle& style = slice_style(latency_slice);
  segments_svg << "<rect class=\"" << style.css_class << "\" x=\""
               << scale.x_for(started_microseconds) << "\" y=\""
               << GanttScale::lane_top(lane_index) + 2 << "\" width=\""
               << segment_width << "\" height=\"" << LANE_HEIGHT_PIXELS - 4
               << "\"><title>" << escape_html_text(node_caption) << " — "
               << style.short_label << " "
               << escape_html_text(format_microseconds(duration_microseconds))
               << "</title></rect>";
}

std::string node_caption(const NodeTimeline& node_timeline) {
  return node_timeline.node->label + " " + node_timeline.node->op_name +
      " (stream " + std::to_string(node_timeline.node->stream_index) + ")";
}

// Lane 0 is the host thread that issues every node; streams follow in index
// order so the chart reads top-down like the streams table below it.
std::map<int, int> build_stream_index_to_lane(const GpuTimeline& gpu_timeline) {
  std::map<int, int> stream_index_to_lane;
  int next_lane_index = HOST_LANE_INDEX + 1;
  for (const StreamTimeline& stream_timeline : gpu_timeline.streams) {
    stream_index_to_lane[stream_timeline.stream_index] = next_lane_index++;
  }
  return stream_index_to_lane;
}

void append_lane(
    std::ostringstream& lanes_svg, int lane_index, const std::string& caption) {
  const double lane_top = GanttScale::lane_top(lane_index);
  lanes_svg << "<rect class=\"lane\" x=\"" << LABEL_GUTTER_PIXELS << "\" y=\""
            << lane_top << "\" width=\"" << PLOT_WIDTH_PIXELS << "\" height=\""
            << LANE_HEIGHT_PIXELS << "\"/><text x=\"8\" y=\""
            << lane_top + LANE_HEIGHT_PIXELS * 0.68 << "\">"
            << escape_html_text(caption) << "</text>";
}

std::string render_lane_backgrounds(
    const GpuTimeline& gpu_timeline,
    const std::map<int, int>& stream_index_to_lane) {
  std::ostringstream lanes_svg;
  append_lane(lanes_svg, HOST_LANE_INDEX, "host (issue)");
  for (const StreamTimeline& stream_timeline : gpu_timeline.streams) {
    append_lane(lanes_svg, stream_index_to_lane.at(stream_timeline.stream_index),
                "stream " + std::to_string(stream_timeline.stream_index) +
                    (stream_timeline.is_gpu ? " (gpu)" : " (cpu)"));
  }
  return lanes_svg.str();
}

// Queued time is not drawn on the stream lane: queued windows of successive
// nodes overlap each other, and the lane shows what the stream was DOING.
// Blank lane = the stream had nothing runnable (idle).
std::string render_node_segments(
    const GpuTimeline& gpu_timeline,
    const GanttScale& scale,
    const std::map<int, int>& stream_index_to_lane) {
  std::ostringstream segments_svg;
  for (const NodeTimeline& node_timeline : gpu_timeline.nodes) {
    const SliceInstants slice_instants = compute_slice_instants(node_timeline);
    const std::string caption = node_caption(node_timeline);
    const int stream_lane =
        stream_index_to_lane.at(node_timeline.node->stream_index);
    for (std::size_t slice_position = 0;
         slice_position < CAUSAL_SLICE_ORDER.size(); ++slice_position) {
      const LatencySlice latency_slice = CAUSAL_SLICE_ORDER[slice_position];
      if (latency_slice == LatencySlice::wait_on_gpu ||
          latency_slice == LatencySlice::completion_latency) {
        continue;
      }
      const int lane_index = latency_slice == LatencySlice::wait_on_cpu
          ? HOST_LANE_INDEX
          : stream_lane;
      append_segment(segments_svg, scale, lane_index,
                     slice_instants[slice_position],
                     slice_instants[slice_position + 1], latency_slice, caption);
    }
  }
  return segments_svg.str();
}

// Arrow from where a producer's GPU work ended to where its consumer's data
// wait ended — the cross-stream edge, drawn where the time was lost.
std::string render_dependency_arrows(
    const GpuTimeline& gpu_timeline,
    const GanttScale& scale,
    const std::map<int, int>& stream_index_to_lane) {
  const std::size_t stalled_end_index =
      slice_end_index(LatencySlice::wait_on_dependency);
  const std::size_t compute_end_index = slice_end_index(LatencySlice::compute);
  std::ostringstream arrows_svg;
  for (const NodeTimeline& consumer : gpu_timeline.nodes) {
    const SliceInstants consumer_instants = compute_slice_instants(consumer);
    const double consumer_y =
        GanttScale::lane_top(stream_index_to_lane.at(consumer.node->stream_index)) +
        LANE_HEIGHT_PIXELS / 2;
    for (const int producer_position : consumer.cross_stream_producer_positions) {
      const NodeTimeline& producer = gpu_timeline.nodes[producer_position];
      const double producer_y =
          GanttScale::lane_top(stream_index_to_lane.at(producer.node->stream_index)) +
          LANE_HEIGHT_PIXELS / 2;
      arrows_svg << "<line class=\"dependency-arrow\" x1=\""
                 << scale.x_for(compute_slice_instants(producer)[compute_end_index])
                 << "\" y1=\"" << producer_y << "\" x2=\""
                 << scale.x_for(consumer_instants[stalled_end_index])
                 << "\" y2=\"" << consumer_y
                 << "\" marker-end=\"url(#dependency-arrowhead)\"><title>"
                 << escape_html_text(node_caption(producer) + " → " +
                                     node_caption(consumer))
                 << "</title></line>";
    }
  }
  return arrows_svg.str();
}


std::string render_stream_table(const GpuTimeline& gpu_timeline) {
  std::ostringstream table_html;
  table_html << "<div class=\"table-wrap\"><table class=\"sortable\"><thead><tr>"
             << "<th>Stream</th><th>Device</th><th>Nodes</th><th>Wall</th>"
             << "<th title=\"Sum of compute slices\">Busy</th>"
             << "<th title=\"Sum of stalled slices\">Stalled</th>"
             << "<th title=\"Wall − busy\">Not computing</th>"
             << "<th>Utilization</th></tr></thead><tbody>";
  for (const StreamTimeline& stream_timeline : gpu_timeline.streams) {
    const double utilization_percent = stream_timeline.wall_microseconds > 0.0
        ? 100.0 * stream_timeline.busy_microseconds /
              stream_timeline.wall_microseconds
        : 0.0;
    char utilization_text[16];
    std::snprintf(utilization_text, sizeof(utilization_text), "%.1f%%",
                  utilization_percent);
    table_html << "<tr><td>" << stream_timeline.stream_index << "</td><td>"
               << (stream_timeline.is_gpu ? "gpu" : "cpu") << "</td><td>"
               << stream_timeline.node_count << "</td><td>"
               << format_microseconds(stream_timeline.wall_microseconds)
               << "</td><td>"
               << format_microseconds(stream_timeline.busy_microseconds)
               << "</td><td>"
               << format_microseconds(stream_timeline.dependency_wait_microseconds)
               << "</td><td>"
               << format_microseconds(stream_timeline.wall_microseconds -
                                      stream_timeline.busy_microseconds)
               << "</td><td data-sort-value=\"" << utilization_percent << "\">"
               << utilization_text << "</td></tr>";
  }
  table_html << "</tbody></table></div>";
  return table_html.str();
}

std::string render_gantt_svg(const GpuTimeline& gpu_timeline) {
  const std::map<int, int> stream_index_to_lane =
      build_stream_index_to_lane(gpu_timeline);
  GanttScale scale;
  scale.total_microseconds = std::max(1.0, gpu_timeline.timeline_wall_microseconds);
  const int lane_count = static_cast<int>(gpu_timeline.streams.size()) + 1;
  const double chart_height = GanttScale::lane_top(lane_count);
  std::ostringstream gantt_svg;
  gantt_svg << "<svg class=\"gantt\" xmlns=\"http://www.w3.org/2000/svg\" width=\""
            << LABEL_GUTTER_PIXELS + PLOT_WIDTH_PIXELS + 16 << "\" height=\""
            << chart_height << "\" role=\"img\" aria-label=\"Per-stream"
            << " timeline\"><defs><marker id=\"dependency-arrowhead\""
            << " viewBox=\"0 0 8 8\" refX=\"7\" refY=\"4\" markerWidth=\"6\""
            << " markerHeight=\"6\" orient=\"auto\"><path d=\"M0,0 L8,4 L0,8 z\""
            << " class=\"slice-stalled\"/></marker></defs>"
            << render_lane_backgrounds(gpu_timeline, stream_index_to_lane)
            << render_axis(scale, chart_height)
            << render_node_segments(gpu_timeline, scale, stream_index_to_lane)
            << render_dependency_arrows(gpu_timeline, scale, stream_index_to_lane)
            << "</svg>";
  return gantt_svg.str();
}

} // namespace

std::string render_gantt_section(const ReportInputs& report_inputs) {
  std::ostringstream section_html;
  section_html << "<section id=\"timeline\"><h2>Timeline (per stream)</h2>"
               << render_slice_legend()
               << "<p class=\"note\">Wall-clock of the timeline pass. The host"
               << " lane shows each node being issued; each stream lane shows"
               << " what that stream was doing: red = stalled on another"
               << " stream's data, amber = GPU scheduling, green = measured GPU"
               << " execution. Blank = nothing runnable on that stream. Arrows"
               << " run from a producer's GPU end to where its cross-stream"
               << " consumer stopped waiting. Hover any segment for details.</p>"
               << "<div class=\"gantt-wrap\">"
               << render_gantt_svg(report_inputs.gpu_timeline) << "</div>"
               << "<h2>Streams</h2>"
               << render_stream_table(report_inputs.gpu_timeline) << "</section>";
  return section_html.str();
}

} // namespace mlx_plan
