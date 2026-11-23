#include "report_writer.h"

#include "dot_writer.h"
#include "svg_renderer.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace mlx_plan {
namespace {

std::string format_microseconds(double duration_microseconds) {
  std::ostringstream duration_text;
  if (duration_microseconds >= 1000.0) {
    duration_text << std::fixed << std::setprecision(2)
                  << duration_microseconds / 1000.0 << " ms";
  } else {
    duration_text << std::fixed << std::setprecision(1) << duration_microseconds
                  << " µs";
  }
  return duration_text.str();
}

std::string format_bytes(size_t byte_count) {
  std::ostringstream bytes_text;
  bytes_text << std::fixed << std::setprecision(1);
  if (byte_count >= 1024 * 1024) {
    bytes_text << byte_count / (1024.0 * 1024.0) << " MiB";
  } else if (byte_count >= 1024) {
    bytes_text << byte_count / 1024.0 << " KiB";
  } else {
    bytes_text << byte_count << " B";
  }
  return bytes_text.str();
}

std::string escape_html_text(const std::string& raw_text) {
  std::string escaped_text;
  for (const char letter : raw_text) {
    switch (letter) {
    case '&':
      escaped_text += "&amp;";
      break;
    case '<':
      escaped_text += "&lt;";
      break;
    case '>':
      escaped_text += "&gt;";
      break;
    case '"':
      escaped_text += "&quot;";
      break;
    default:
      escaped_text += letter;
    }
  }
  return escaped_text;
}

} // namespace

std::string build_summary_cards(
    const CapturedGraph& captured_graph,
    const RunTelemetry& run_telemetry) {
  std::ostringstream cards_html;
  cards_html << "<div class=\"cards\">\n";
  cards_html << "  <div class=\"card\"><div class=\"card-value\">"
             << format_microseconds(run_telemetry.pipeline_total_microseconds)
             << "</div><div class=\"card-label\">Pipeline wall-clock"
             << " (uninstrumented pass)</div></div>\n";
  cards_html << "  <div class=\"card\"><div class=\"card-value\">"
             << format_microseconds(run_telemetry.attributed_total_microseconds)
             << "</div><div class=\"card-label\">Sum of node durations"
             << " (attributed pass)</div></div>\n";
  cards_html << "  <div class=\"card\"><div class=\"card-value\">"
             << format_bytes(run_telemetry.run_peak_memory_bytes)
             << "</div><div class=\"card-label\">Peak memory</div></div>\n";
  cards_html << "  <div class=\"card\"><div class=\"card-value\">"
             << run_telemetry.node_count
             << "</div><div class=\"card-label\">Primitive nodes</div></div>\n";
  cards_html << "  <div class=\"card\"><div class=\"card-value\">"
             << captured_graph.stream_count
             << "</div><div class=\"card-label\">Streams"
             << " (stage parallelism)</div></div>\n";
  cards_html << "</div>\n";
  return cards_html.str();
}

std::string build_stage_table(const RunTelemetry& run_telemetry) {
  std::ostringstream table_html;
  table_html << "<table>\n";
  table_html << "  <tr><th>#</th><th>Op</th><th>Node</th><th>Stream</th>"
             << "<th>Shape</th><th>Output</th><th>Duration</th>"
             << "<th>Memory added</th></tr>\n";

  double longest_duration_microseconds = 1.0;
  for (const NodeTelemetry& node_telemetry : run_telemetry.attributed_nodes) {
    longest_duration_microseconds = std::max(
        longest_duration_microseconds, node_telemetry.duration_microseconds);
  }

  for (size_t node_position = 0; node_position < run_telemetry.attributed_nodes.size();
       ++node_position) {
    const NodeTelemetry& node_telemetry = run_telemetry.attributed_nodes[node_position];
    const double duration_share =
        node_telemetry.duration_microseconds / longest_duration_microseconds;
    table_html << "  <tr>"
               << "<td>" << node_position << "</td>"
               << "<td><b>" << escape_html_text(node_telemetry.node->op_name)
               << "</b></td>"
               << "<td>" << escape_html_text(node_telemetry.node->label) << "</td>"
               << "<td><span class=\"stream-badge\">stream "
               << node_telemetry.node->stream_index << "</span></td>"
               << "<td>" << escape_html_text(node_telemetry.node->shape_string)
               << " " << escape_html_text(node_telemetry.node->dtype_string)
               << "</td>"
               << "<td>" << format_bytes(node_telemetry.output_bytes) << "</td>"
               << "<td><div class=\"duration-bar\" style=\"width:"
               << static_cast<int>(duration_share * 100.0) << "%\">"
               << format_microseconds(node_telemetry.duration_microseconds)
               << "</div></td>"
               << "<td>" << format_bytes(node_telemetry.memory_added_bytes)
               << "</td></tr>\n";
  }
  table_html << "</table>\n";
  return table_html.str();
}

std::string build_report_document(
    const CapturedGraph& captured_graph,
    const RunTelemetry& run_telemetry,
    const ReportPaths& report_paths,
    const std::string& plan_svg,
    const std::string& plan_dot,
    const std::string& report_title) {
  std::ostringstream report_html;
  report_html << "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\">\n"
              << "<title>" << escape_html_text(report_title) << "</title>\n"
              << "<style>\n"
              << "  body { font-family: Helvetica, Arial, sans-serif;"
              << " margin: 0; color: #333; }\n"
              << "  header { background: #1b2a3a; color: #fff; padding:"
              << " 14px 24px; }\n"
              << "  header h1 { margin: 0; font-size: 20px; }\n"
              << "  header p { margin: 2px 0 0; color: #9fb4c8;"
              << " font-size: 12px; }\n"
              << "  section { padding: 16px 24px; }\n"
              << "  h2 { font-size: 15px; color: #1b2a3a;"
              << " border-bottom: 1px solid #d8dee5; padding-bottom: 6px; }\n"
              << "  .cards { display: flex; gap: 12px; flex-wrap: wrap; }\n"
              << "  .card { background: #f4f7fa; border: 1px solid #d8dee5;"
              << " border-radius: 6px; padding: 10px 16px; min-width: 150px; }\n"
              << "  .card-value { font-size: 22px; font-weight: 700;"
              << " color: #1b2a3a; }\n"
              << "  .card-label { font-size: 11px; color: #66788a;"
              << " margin-top: 2px; }\n"
              << "  table { border-collapse: collapse; width: 100%;"
              << " font-size: 12px; }\n"
              << "  th, td { border: 1px solid #d8dee5; padding: 5px 8px;"
              << " text-align: left; }\n"
              << "  th { background: #eef2f6; }\n"
              << "  .duration-bar { background: #2f6fab; color: #fff;"
              << " border-radius: 3px; padding: 2px 6px;"
              << " min-width: 60px; }\n"
              << "  .stream-badge { background: #1b2a3a; color: #fff;"
              << " border-radius: 3px; padding: 2px 7px; font-size: 11px; }\n"
              << "  .dag svg { max-width: 100%; height: auto; }\n"
              << "  details summary { cursor: pointer; color: #2f6fab; }\n"
              << "  pre { background: #f4f7fa; border: 1px solid #d8dee5;"
              << " padding: 10px; overflow: auto; font-size: 11px; }\n"
              << "</style></head><body>\n"
              << "<header><h1>" << escape_html_text(report_title)
              << "</h1><p>mlx-visual-plan — Spark-UI-style report for one MLX"
              << " computation</p></header>\n"
              << "<section>" << build_summary_cards(captured_graph, run_telemetry)
              << "</section>\n"
              << "<section><h2>Execution DAG</h2><div class=\"dag\">";
  if (!plan_svg.empty()) {
    report_html << plan_svg;
  } else {
    report_html << "<p>Graphviz `dot` was unavailable; the raw DOT is below"
                << " (and at " << escape_html_text(report_paths.dot_path)
                << ").</p><pre>" << escape_html_text(plan_dot) << "</pre>";
  }
  report_html << "</div></section>\n"
              << "<section><h2>Stage table (per primitive node)</h2>"
              << build_stage_table(run_telemetry)
              << "</section>\n"
              << "<section><h2>Raw plan (Graphviz DOT)</h2><details>"
              << "<summary>Show DOT source</summary><pre>"
              << escape_html_text(plan_dot) << "</pre></details>"
              << "<p style=\"font-size:11px;color:#66788a\">Attribution"
              << " serializes the pipeline (each node waits on its stream),"
              << " so node durations measure kernels without overlap; the"
              << " pipeline wall-clock above is the honest end-to-end"
              << " number.</p></section>\n"
              << "</body></html>\n";
  return report_html.str();
}

bool write_text_file(const std::string& file_path, const std::string& file_text) {
  std::ofstream output_stream(file_path, std::ios::binary);
  return output_stream.good() &&
         output_stream.write(
             file_text.data(), static_cast<std::streamsize>(file_text.size()))
             .good();
}

ReportPaths write_report(
    const CapturedGraph& captured_graph,
    const RunTelemetry& run_telemetry,
    const std::string& output_directory,
    const std::string& plan_file_base,
    const std::string& report_title) {
  ReportPaths report_paths;
  report_paths.html_path = output_directory + "/" + plan_file_base + ".html";
  report_paths.dot_path = output_directory + "/" + plan_file_base + ".dot";

  const std::string plan_dot = write_plan_dot(
      captured_graph, run_telemetry, report_title);
  const std::string plan_svg =
      render_plan_svg(plan_dot, output_directory, plan_file_base);
  report_paths.graphviz_available = !plan_svg.empty();
  if (report_paths.graphviz_available) {
    report_paths.svg_path = output_directory + "/" + plan_file_base + ".svg";
  }

  const std::string report_html = build_report_document(
      captured_graph,
      run_telemetry,
      report_paths,
      plan_svg,
      plan_dot,
      report_title);
  write_text_file(report_paths.html_path, report_html);
  write_text_file(report_paths.dot_path, plan_dot);
  return report_paths;
}

} // namespace mlx_plan