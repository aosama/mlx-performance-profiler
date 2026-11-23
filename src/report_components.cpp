#include "report_components.h"

#include "latency_slices.h"
#include "text_format.h"

#include <sstream>

namespace mlx_plan {

std::string render_stacked_bar(const LatencyBreakdown& breakdown) {
  std::ostringstream bar_html;
  bar_html << "<div class=\"stacked-bar\">";
  if (breakdown.wall_microseconds > 0.0) {
    for (const LatencySlice latency_slice : CAUSAL_SLICE_ORDER) {
      const double slice_duration = slice_microseconds(breakdown, latency_slice);
      if (slice_duration <= 0.0) {
        continue;
      }
      const SliceStyle& style = slice_style(latency_slice);
      bar_html << "<span class=\"" << style.css_class << "\" style=\"width:"
               << 100.0 * slice_duration / breakdown.wall_microseconds
               << "%\" title=\"" << style.short_label << ": "
               << escape_html_text(format_microseconds(slice_duration))
               << "\"></span>";
    }
  }
  bar_html << "</div>";
  return bar_html.str();
}

std::string render_slice_legend() {
  std::ostringstream legend_html;
  legend_html << "<div class=\"legend\">";
  for (const LatencySlice latency_slice : CAUSAL_SLICE_ORDER) {
    const SliceStyle& style = slice_style(latency_slice);
    legend_html << "<span title=\"" << escape_html_text(style.meaning)
                << "\"><i class=\"swatch " << style.css_class << "\"></i>"
                << style.short_label << " <small>(" << style.spark_analog
                << ")</small></span>";
  }
  legend_html << "</div>";
  return legend_html.str();
}

std::string render_class_badge(NodeClass node_class) {
  const std::string_view class_name = node_class_name(node_class);
  std::ostringstream badge_html;
  badge_html << "<span class=\"class-badge "
             << slice_style(slice_for_class(node_class)).css_class << "\">"
             << class_name << "</span>";
  return badge_html.str();
}

} // namespace mlx_plan
