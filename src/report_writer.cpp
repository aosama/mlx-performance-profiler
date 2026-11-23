#include "report_writer.h"

#include "dot_writer.h"
#include "report_assets.h"
#include "report_gantt_section.h"
#include "report_summary_section.h"
#include "report_task_table_section.h"
#include "svg_renderer.h"
#include "text_format.h"

#include <fstream>
#include <sstream>

namespace mlx_plan {
namespace {

std::string render_page_header(const std::string& report_title) {
  std::ostringstream header_html;
  header_html << "<header><h1>" << escape_html_text(report_title)
              << "</h1><p>mlx-visual-plan — what MLX actually did, and what it"
              << " was waiting for</p></header><nav><a href=\"#job\">Job</a>"
              << "<a href=\"#timeline\">Timeline</a><a href=\"#tasks\">Tasks</a>"
              << "<a href=\"#dag\">DAG</a><a href=\"#method\">Method</a></nav>";
  return header_html.str();
}

std::string render_dag_section(const std::string& plan_svg, const std::string& plan_dot) {
  std::ostringstream section_html;
  section_html << "<section id=\"dag\"><h2>Execution DAG</h2><p class=\"note\">"
               << "Fill = stream; red outline = node classified as stalled on"
               << " another stream; edge labels = bytes read.</p><div class=\"dag\">";
  if (!plan_svg.empty()) {
    section_html << plan_svg;
  } else {
    section_html << "<p class=\"note\">Graphviz <code>dot</code> was unavailable;"
                 << " the raw DOT follows.</p><pre>" << escape_html_text(plan_dot)
                 << "</pre>";
  }
  section_html << "</div><details><summary>DOT source</summary><pre>"
               << escape_html_text(plan_dot) << "</pre></details></section>";
  return section_html.str();
}

std::string render_method_section() {
  return "<section id=\"method\"><h2>How these numbers were measured</h2>"
         "<p class=\"note\">Four passes run on identical, freshly built graphs."
         " <b>Warm-up</b> absorbs Metal kernel compilation and is discarded."
         " <b>Job duration</b> is one uninstrumented eval. The <b>timeline</b>"
         " pass issues one <code>async_eval</code> per node (streams keep"
         " overlapping) and reads each node's Metal command-buffer"
         " <code>GPUStartTime/GPUEndTime</code>: compute is GPU-measured at"
         " command-buffer granularity, and producers count as ready at their GPU"
         " end. Its per-node command buffers add submission cost a plain eval"
         " batches away — hence the gap between instrumented and job duration."
         " <b>Sum of stages</b> times each node alone. Never read one of these"
         " as the whole truth; that is why all are shown.</p></section>";
}

bool write_text_file(const std::string& file_path, const std::string& file_text) {
  std::ofstream output_stream(file_path, std::ios::binary);
  return output_stream.good() &&
      output_stream
          .write(file_text.data(), static_cast<std::streamsize>(file_text.size()))
          .good();
}

} // namespace

std::string build_report_document(
    const ReportInputs& report_inputs,
    const std::string& plan_svg,
    const std::string& plan_dot) {
  std::ostringstream report_html;
  report_html << "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
              << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
              << "<title>" << escape_html_text(report_inputs.report_title)
              << "</title><style>" << report_stylesheet() << "</style></head><body>"
              << render_page_header(report_inputs.report_title)
              << render_summary_section(report_inputs)
              << render_gantt_section(report_inputs)
              << render_task_table_section(report_inputs)
              << render_dag_section(plan_svg, plan_dot) << render_method_section()
              << "<script>" << report_sort_script() << "</script></body></html>\n";
  return report_html.str();
}

ReportPaths write_report(
    const ReportInputs& report_inputs,
    const std::string& output_directory,
    const std::string& plan_file_base) {
  ReportPaths report_paths;
  report_paths.html_path = output_directory + "/" + plan_file_base + ".html";
  report_paths.dot_path = output_directory + "/" + plan_file_base + ".dot";
  const std::string plan_dot = write_plan_dot(
      report_inputs.timeline_graph, report_inputs.gpu_timeline,
      report_inputs.report_title);
  const std::string plan_svg =
      render_plan_svg(plan_dot, output_directory, plan_file_base);
  report_paths.graphviz_available = !plan_svg.empty();
  if (report_paths.graphviz_available) {
    report_paths.svg_path = output_directory + "/" + plan_file_base + ".svg";
  }
  write_text_file(report_paths.html_path,
                  build_report_document(report_inputs, plan_svg, plan_dot));
  write_text_file(report_paths.dot_path, plan_dot);
  return report_paths;
}

} // namespace mlx_plan
