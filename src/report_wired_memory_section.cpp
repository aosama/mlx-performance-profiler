#include "report_wired_memory_section.h"

#include "text_format.h"
#include "wired_memory_analysis.h"

#include <algorithm>
#include <sstream>

namespace mlx_perf {
namespace {

constexpr double CHART_WIDTH = 1000.0;
constexpr double CHART_HEIGHT = 240.0;
constexpr double PLOT_LEFT = 78.0;
constexpr double PLOT_RIGHT = 990.0;
constexpr double PLOT_TOP = 12.0;
constexpr double PLOT_BOTTOM = 212.0;
// Draw the limit as a line only when it sits near the data; a limit far
// above (tens of GiB over a few GiB wired) would flatten every series.
constexpr double LIMIT_ON_CHART_FACTOR = 1.5;

struct ChartScale {
  double first_sample_seconds = 0.0;
  double span_seconds = 1e-9;
  double top_bytes = 1.0;
  double x_of(double host_seconds) const {
    return PLOT_LEFT + (host_seconds - first_sample_seconds) / span_seconds * (PLOT_RIGHT - PLOT_LEFT);
  }
  double y_of(double byte_count) const {
    return PLOT_BOTTOM - std::max(0.0, byte_count) / top_bytes * (PLOT_BOTTOM - PLOT_TOP);
  }
};

bool is_limit_on_chart(const WiredMemoryAnalysis& wired_memory_analysis) {
  const double highest_bytes = static_cast<double>(
      std::max(wired_memory_analysis.peak_system_wired_bytes, wired_memory_analysis.peak_mlx_held_bytes));
  return wired_memory_analysis.wire_limit_bytes > 0 &&
         static_cast<double>(wired_memory_analysis.wire_limit_bytes) <= highest_bytes * LIMIT_ON_CHART_FACTOR;
}

ChartScale make_scale(const std::vector<WiredMemorySample>& wired_memory_samples,
                      const WiredMemoryAnalysis& wired_memory_analysis) {
  ChartScale chart_scale;
  chart_scale.first_sample_seconds = wired_memory_samples.front().host_seconds;
  chart_scale.span_seconds =
      std::max(wired_memory_samples.back().host_seconds - chart_scale.first_sample_seconds, 1e-9);
  double top_bytes = static_cast<double>(
      std::max(wired_memory_analysis.peak_system_wired_bytes, wired_memory_analysis.peak_mlx_held_bytes));
  if (is_limit_on_chart(wired_memory_analysis)) {
    top_bytes = std::max(top_bytes, static_cast<double>(wired_memory_analysis.wire_limit_bytes));
  }
  chart_scale.top_bytes = std::max(top_bytes * 1.08, 1.0);
  return chart_scale;
}

std::string render_axes(const ChartScale& chart_scale) {
  std::ostringstream axes_svg;
  for (int tick_index = 0; tick_index <= 4; ++tick_index) {
    const double tick_bytes = chart_scale.top_bytes * tick_index / 4.0;
    const double tick_y = chart_scale.y_of(tick_bytes);
    axes_svg << "<line class=\"grid\" x1=\"" << PLOT_LEFT << "\" x2=\"" << PLOT_RIGHT << "\" y1=\"" << tick_y
             << "\" y2=\"" << tick_y << "\"/><text class=\"axis\" x=\"" << PLOT_LEFT - 6 << "\" y=\""
             << tick_y + 4 << "\" text-anchor=\"end\">" << format_bytes(static_cast<std::size_t>(tick_bytes))
             << "</text>";
    const double tick_seconds = chart_scale.first_sample_seconds + chart_scale.span_seconds * tick_index / 4.0;
    // The end labels anchor inward, so neither is cut off at the plot's edge.
    const char* text_anchor = tick_index == 0 ? "start" : tick_index == 4 ? "end" : "middle";
    axes_svg << "<text class=\"axis\" x=\"" << chart_scale.x_of(tick_seconds) << "\" y=\"" << CHART_HEIGHT - 6
             << "\" text-anchor=\"" << text_anchor << "\">"
             << format_microseconds((tick_seconds - chart_scale.first_sample_seconds) * 1e6) << "</text>";
  }
  return axes_svg.str();
}

enum class Series { system_wired, mlx_held, outside_mlx };

double series_bytes(const WiredMemorySample& wired_memory_sample, Series series) {
  if (series == Series::system_wired) return static_cast<double>(wired_memory_sample.system_wired_bytes);
  if (series == Series::mlx_held) return static_cast<double>(mlx_held_bytes(wired_memory_sample));
  return static_cast<double>(wired_outside_mlx_bytes(wired_memory_sample));
}

std::string render_series(const std::vector<WiredMemorySample>& wired_memory_samples, const ChartScale& chart_scale,
                          Series series, const char* css_class) {
  std::ostringstream series_svg;
  series_svg << "<polyline class=\"" << css_class << "\" points=\"";
  for (const WiredMemorySample& wired_memory_sample : wired_memory_samples) {
    series_svg << chart_scale.x_of(wired_memory_sample.host_seconds) << ','
               << chart_scale.y_of(series_bytes(wired_memory_sample, series)) << ' ';
  }
  series_svg << "\"/>";
  return series_svg.str();
}

std::string render_boundary_ticks(const std::vector<WiredMemorySample>& wired_memory_samples,
                                  const ChartScale& chart_scale) {
  std::ostringstream ticks_svg;
  for (const WiredMemorySample& wired_memory_sample : wired_memory_samples) {
    if (wired_memory_sample.sample_point != WiredSamplePoint::evaluation_boundary) continue;
    const double tick_x = chart_scale.x_of(wired_memory_sample.host_seconds);
    ticks_svg << "<line class=\"boundary\" x1=\"" << tick_x << "\" x2=\"" << tick_x << "\" y1=\""
              << PLOT_BOTTOM << "\" y2=\"" << PLOT_BOTTOM + 7 << "\"/>";
  }
  return ticks_svg.str();
}

std::string render_marks(const std::vector<WiredMemorySample>& wired_memory_samples, const ChartScale& chart_scale,
                         const std::vector<int>& sample_indices, const char* css_class, double radius) {
  std::ostringstream marks_svg;
  for (const int sample_index : sample_indices) {
    const WiredMemorySample& marked_sample = wired_memory_samples[sample_index];
    marks_svg << "<circle class=\"" << css_class << "\" r=\"" << radius << "\" cx=\""
              << chart_scale.x_of(marked_sample.host_seconds) << "\" cy=\""
              << chart_scale.y_of(series_bytes(marked_sample, Series::outside_mlx)) << "\"/>";
  }
  return marks_svg.str();
}

std::string render_release_events(const std::vector<WiredMemorySample>& wired_memory_samples,
                                  const ChartScale& chart_scale, const WiredMemoryAnalysis& wired_memory_analysis) {
  std::ostringstream events_svg;
  for (const WireReleaseEvent& release_event : wired_memory_analysis.release_events) {
    const double released_x = chart_scale.x_of(wired_memory_samples[release_event.released_sample_index].host_seconds);
    events_svg << "<line class=\"release\" x1=\"" << released_x << "\" x2=\"" << released_x << "\" y1=\""
               << PLOT_TOP << "\" y2=\"" << PLOT_BOTTOM << "\"/>";
    if (!release_event.unwired_sample_index.has_value()) continue;
    const double unwired_x = chart_scale.x_of(wired_memory_samples[*release_event.unwired_sample_index].host_seconds);
    events_svg << "<line class=\"unwired\" x1=\"" << unwired_x << "\" x2=\"" << unwired_x << "\" y1=\""
               << PLOT_TOP << "\" y2=\"" << PLOT_BOTTOM << "\"/><text class=\"axis\" x=\""
               << (released_x + unwired_x) / 2 << "\" y=\"" << PLOT_TOP + 12 << "\" text-anchor=\"middle\">"
               << "un-wired after " << format_microseconds(*release_event.release_lag_microseconds) << "</text>";
  }
  return events_svg.str();
}

std::string render_limit_line(const ChartScale& chart_scale, const WiredMemoryAnalysis& wired_memory_analysis) {
  if (!is_limit_on_chart(wired_memory_analysis)) return "";
  const double limit_y = chart_scale.y_of(static_cast<double>(wired_memory_analysis.wire_limit_bytes));
  std::ostringstream limit_svg;
  limit_svg << "<line class=\"limit\" x1=\"" << PLOT_LEFT << "\" x2=\"" << PLOT_RIGHT << "\" y1=\"" << limit_y
            << "\" y2=\"" << limit_y << "\"/><text class=\"axis\" x=\"" << PLOT_RIGHT << "\" y=\"" << limit_y - 4
            << "\" text-anchor=\"end\">wire limit</text>";
  return limit_svg.str();
}

std::string render_chart(const std::vector<WiredMemorySample>& wired_memory_samples,
                         const WiredMemoryAnalysis& wired_memory_analysis) {
  const ChartScale chart_scale = make_scale(wired_memory_samples, wired_memory_analysis);
  std::ostringstream chart_svg;
  chart_svg << "<svg class=\"wired-chart\" viewBox=\"0 0 " << CHART_WIDTH << ' ' << CHART_HEIGHT
            << "\" role=\"img\" aria-label=\"Wired memory over the run\">" << render_axes(chart_scale)
            << render_limit_line(chart_scale, wired_memory_analysis)
            << render_release_events(wired_memory_samples, chart_scale, wired_memory_analysis)
            << render_series(wired_memory_samples, chart_scale, Series::system_wired, "system-wired")
            << render_series(wired_memory_samples, chart_scale, Series::mlx_held, "mlx-held")
            << render_series(wired_memory_samples, chart_scale, Series::outside_mlx, "outside-mlx")
            << render_marks(wired_memory_samples, chart_scale,
                            wired_memory_analysis.release_lag_inflated_sample_indices, "pending-unwire", 2.2)
            << render_marks(wired_memory_samples, chart_scale,
                            wired_memory_analysis.transient_jump_sample_indices, "transient", 5.0)
            << render_boundary_ticks(wired_memory_samples, chart_scale) << "</svg>";
  return chart_svg.str();
}

std::string bytes_or_dash(const std::optional<std::int64_t>& byte_count) {
  if (!byte_count.has_value()) return "—";
  const std::string magnitude = format_bytes(static_cast<std::size_t>(std::llabs(*byte_count)));
  return *byte_count < 0 ? "−" + magnitude : magnitude;
}

std::string describe_release_lag(const WiredMemoryAnalysis& wired_memory_analysis) {
  const std::vector<WireReleaseEvent>& release_events = wired_memory_analysis.release_events;
  if (release_events.empty()) return "no release of " + format_bytes(RELEASE_EVENT_MINIMUM_BYTES) + " or more";
  const std::optional<double> median_lag = median_release_lag_microseconds(release_events);
  const std::optional<double> largest_lag = largest_release_lag_microseconds(release_events);
  const auto unresolved_count = std::count_if(release_events.begin(), release_events.end(),
                                              [](const WireReleaseEvent& release_event) {
                                                return !release_event.unwired_sample_index.has_value();
                                              });
  std::ostringstream lag_text;
  lag_text << release_events.size() << (release_events.size() == 1 ? " release" : " releases");
  if (median_lag.has_value()) {
    lag_text << "; un-wired after " << format_microseconds(*median_lag) << " (median), "
             << format_microseconds(*largest_lag) << " at most";
  }
  if (unresolved_count > 0) lag_text << "; " << unresolved_count << " not un-wired before sampling stopped";
  return lag_text.str();
}

std::string summary_row(const std::string& figure_name, const std::string& figure_value) {
  return "<tr><td>" + escape_html_text(figure_name) + "</td><td class=\"number\">" + escape_html_text(figure_value) +
         "</td></tr>";
}

std::string render_summary(const WiredMemoryAnalysis& wired_memory_analysis) {
  const bool is_limit_known = wired_memory_analysis.wire_limit_bytes > 0;
  std::ostringstream summary_html;
  summary_html << "<div class=\"table-wrap\"><table class=\"wired-summary\"><tbody>"
               << summary_row("Wire limit (vm.global_user_wire_limit)",
                              is_limit_known ? format_bytes(wired_memory_analysis.wire_limit_bytes) : "unknown")
               << summary_row("Peak system wired", format_bytes(wired_memory_analysis.peak_system_wired_bytes))
               << summary_row("Peak MLX holding (active + cache)", format_bytes(wired_memory_analysis.peak_mlx_held_bytes))
               << summary_row("Peak wired outside MLX",
                              bytes_or_dash(wired_memory_analysis.peak_wired_outside_mlx_bytes))
               << summary_row("Lowest wired outside MLX",
                              bytes_or_dash(wired_memory_analysis.lowest_wired_outside_mlx_bytes))
               << summary_row("Least room under the limit", bytes_or_dash(wired_memory_analysis.lowest_wire_headroom_bytes))
               << summary_row("Crossed the limit",
                              is_limit_known ? (wired_memory_analysis.has_crossed_wire_limit ? "yes" : "no") : "—")
               << summary_row("MLX releases and macOS un-wiring", describe_release_lag(wired_memory_analysis))
               << summary_row("Samples counting released memory still wired",
                              std::to_string(wired_memory_analysis.release_lag_inflated_sample_indices.size()) +
                                  " (by up to " +
                                  format_bytes(wired_memory_analysis.largest_release_lag_inflation_bytes) + ")")
               << summary_row("Transient jumps in wired outside MLX",
                              std::to_string(wired_memory_analysis.transient_jump_sample_indices.size()))
               << summary_row("Samples", std::to_string(wired_memory_analysis.sample_count))
               << "</tbody></table></div>";
  return summary_html.str();
}

} // namespace

std::string render_wired_memory_section(const WiredMemorySectionText& section_text,
                                        const std::vector<WiredMemorySample>& wired_memory_samples) {
  if (wired_memory_samples.empty()) return "";
  const WiredMemoryAnalysis wired_memory_analysis = analyze_wired_memory(wired_memory_samples);
  std::ostringstream section_html;
  section_html << "<section id=\"" << escape_html_text(section_text.section_id) << "\"><h2>"
               << escape_html_text(section_text.heading) << "</h2>" << render_chart(wired_memory_samples, wired_memory_analysis)
               << "<p class=\"wired-legend\"><span class=\"key system-wired\">system wired</span>"
               << "<span class=\"key mlx-held\">MLX active + cache</span>"
               << "<span class=\"key outside-mlx\">wired outside MLX</span>"
               << "<span class=\"key pending-unwire\">released, not yet un-wired</span>"
               << "<span class=\"key transient\">transient jump</span>"
               << "<span class=\"key boundary\">evaluation boundary</span></p>"
               << render_summary(wired_memory_analysis) << "<p class=\"note\">" << escape_html_text(section_text.source_note)
               << " On Apple silicon macOS wires memory when the GPU uses it and keeps wired memory under the wire"
               << " limit. It un-wires memory MLX released a fraction of a second later (measure this Mac's delay"
               << " with mlx_perf_measure_wire_release_lag), and memory MLX still holds after about a second without"
               << " GPU use, wiring it again on its next use. \"Wired outside MLX\" (the system's wired total less MLX's active memory and cache, the"
               << " figure an engine sizes against) is therefore too high just after a release (those samples are"
               << " marked) and too low while MLX's memory sits idle. A release is a fall of "
               << format_bytes(RELEASE_EVENT_MINIMUM_BYTES) << " or more in MLX's holding between two samples, timed"
               << " from the first sample that saw it.</p></section>";
  return section_html.str();
}

} // namespace mlx_perf
