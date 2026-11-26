#include "report_writer.h"

#include "dot_writer.h"
#include "report_assets.h"
#include "report_gantt_section.h"
#include "report_stage_table_section.h"
#include "report_summary_section.h"
#include "report_task_table_section.h"
#include "svg_renderer.h"
#include "text_format.h"

#include <fstream>
#include <sstream>

namespace mlx_perf {
namespace {

std::string render_page_header(const std::string& report_title) {
  std::ostringstream header_html;
  header_html << "<header><h1>" << escape_html_text(report_title)
              << "</h1><p>mlx-performance-profiler — what MLX actually did, and what it"
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

std::string render_method_section(TimelineGranularity granularity) {
  const std::string stage_note = granularity == TimelineGranularity::per_stage
      ? " <b>This report was timed per stage:</b> the job named its stages (for a"
        " decoder, one per layer), and the timeline pass issued one"
        " <code>async_eval</code> per stage instead of per node, because at this"
        " graph size per-node command buffers would cost several times the job and"
        " every node would read as queued behind the instrumentation. A stage's GPU"
        " window runs from its first command buffer's GPU start to the end of its"
        " last buffer, bounded by an empty marker buffer queued behind it (MLX"
        " commits mid-stage when a buffer exceeds its op budget, and those middle"
        " buffers are not observable) and by host-observed completion."
      : std::string();
  return "<section id=\"method\"><h2>How these numbers were measured</h2>"
         "<p class=\"note\">Four passes run on identical, freshly built graphs."
         " <b>Warm-up</b> absorbs Metal kernel compilation and is discarded."
         " <b>Job duration</b> is the median of several uninstrumented evals. The <b>timeline</b>"
         " pass issues one <code>async_eval</code> per node (streams keep"
         " overlapping) and reads each node's Metal command-buffer"
         " <code>GPUStartTime/GPUEndTime</code>: compute is GPU-measured at"
         " command-buffer granularity, and producers count as ready at their GPU"
         " end. Its per-node command buffers add submission cost a plain eval"
         " batches away — hence the gap between instrumented and job duration."
         " <b>Sum of stages</b> times each node alone. Never read one of these"
         " as the whole truth; that is why all are shown." + stage_note +
      "</p></section>";
}

} // namespace

std::string render_tasks_section(const ReportInputs& report_inputs) {
  return report_inputs.gpu_timeline.granularity == TimelineGranularity::per_stage
      ? render_stage_table_section(report_inputs)
      : render_task_table_section(report_inputs);
}

bool write_text_file(const std::string& file_path, const std::string& file_text) {
  std::ofstream output_stream(file_path, std::ios::binary);
  return output_stream.good() &&
      output_stream
          .write(file_text.data(), static_cast<std::streamsize>(file_text.size()))
          .good();
}

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
              << render_tasks_section(report_inputs)
              << render_dag_section(plan_svg, plan_dot) << render_method_section(report_inputs.gpu_timeline.granularity)
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
  report_paths.plan_svg = plan_svg;
  if (report_paths.graphviz_available) {
    report_paths.svg_path = output_directory + "/" + plan_file_base + ".svg";
  }
  const bool is_html_written = write_text_file(
      report_paths.html_path, build_report_document(report_inputs, plan_svg, plan_dot));
  const bool is_dot_written = write_text_file(report_paths.dot_path, plan_dot);
  report_paths.is_written = is_html_written && is_dot_written;
  return report_paths;
}

} // namespace mlx_perf
