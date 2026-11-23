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

std::string render_number_cell(double sort_value, const std::string& shown) {
  std::ostringstream cell_html;
  cell_html << "<td class=\"number\" data-sort-value=\"" << sort_value << "\">"
            << escape_html_text(shown) << "</td>";
  return cell_html.str();
}

std::string render_duration_cell(double microseconds) {
  return render_number_cell(microseconds, format_microseconds(microseconds));
}

std::string render_slice_header_cells() {
  std::ostringstream header_html;
  for (const LatencySlice latency_slice : CAUSAL_SLICE_ORDER) {
    const SliceStyle& style = slice_style(latency_slice);
    header_html << "<th title=\"" << escape_html_text(style.meaning) << "\">"
                << style.short_label << "</th>";
  }
  return header_html.str();
}

std::string render_slice_cells(const LatencyBreakdown& breakdown) {
  std::string cells_html;
  for (const LatencySlice latency_slice : CAUSAL_SLICE_ORDER) {
    const double slice_duration = slice_microseconds(breakdown, latency_slice);
    const bool is_residual_compute =
        latency_slice == LatencySlice::compute && !breakdown.is_compute_measured;
    cells_html += render_number_cell(
        slice_duration,
        (is_residual_compute ? "~" : "") + format_microseconds(slice_duration));
  }
  return cells_html;
}

} // namespace mlx_plan
