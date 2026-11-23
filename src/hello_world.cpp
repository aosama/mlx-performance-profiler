// The demo: measure one small three-stream job twice — as written (eager) and
// through mlx::core::compile — and write a Spark-style report for each plus
// a side-by-side comparison. Orchestration only: the job lives in
// demo_model, the passes in measurement_session, rendering in report_*.

#include "demo_model.h"
#include "measurement_session.h"
#include "report_comparison_page.h"
#include "report_writer.h"
#include "timeline_console.h"

#include <cstdio>
#include <string>

namespace {

using namespace mlx_plan;

const std::string OUTPUT_DIRECTORY = "build";
const std::string JOB_TITLE = "Hello World — three-stream toy job";

void print_variant_console(const VariantMeasurement& variant_measurement) {
  std::printf("\n######## %s ########", variant_measurement.variant_name.c_str());
  print_job_summary(variant_measurement.job_totals, variant_measurement.gpu_timeline);
  print_stream_table(variant_measurement.gpu_timeline);
  print_node_table(variant_measurement.gpu_timeline);
}

ReportPaths write_variant_report(
    const VariantMeasurement& variant_measurement, const std::string& file_base) {
  const ReportInputs report_inputs{
      JOB_TITLE + " (" + variant_measurement.variant_name + ")",
      variant_measurement.timeline_graph, variant_measurement.gpu_timeline,
      variant_measurement.serial_telemetry, variant_measurement.job_totals};
  const ReportPaths report_paths =
      write_report(report_inputs, OUTPUT_DIRECTORY, file_base);
  if (!report_paths.is_written) {
    std::fprintf(stderr, "error: could not write %s\n", report_paths.html_path.c_str());
  } else {
    std::printf("report: %s%s\n", report_paths.html_path.c_str(),
                report_paths.graphviz_available ? " (with SVG)" : " (no Graphviz)");
  }
  return report_paths;
}

void write_comparison_report(
    const VariantMeasurement& eager_measurement,
    const ReportPaths& eager_paths,
    const VariantMeasurement& compiled_measurement,
    const ReportPaths& compiled_paths) {
  const ComparedVariant eager_variant{
      eager_measurement, eager_paths.plan_svg, "hello_world_eager.html"};
  const ComparedVariant compiled_variant{
      compiled_measurement, compiled_paths.plan_svg, "hello_world_compiled.html"};
  const std::string comparison_path = OUTPUT_DIRECTORY + "/hello_world.html";
  if (!write_text_file(comparison_path,
                       build_comparison_document(JOB_TITLE + ": eager vs compiled",
                                                 eager_variant, compiled_variant))) {
    std::fprintf(stderr, "error: could not write %s\n", comparison_path.c_str());
    return;
  }
  std::printf("report: %s (eager vs compiled)\n", comparison_path.c_str());
}

} // namespace

int main() {
  const DemoStreams demo_streams = make_demo_streams();
  std::printf("materializing demo inputs...\n");
  const auto demo_inputs = make_materialized_demo_inputs(demo_streams);
  const EagerDemoGraphFactory eager_factory(demo_streams, demo_inputs);
  const CompiledDemoGraphFactory compiled_factory(demo_streams, demo_inputs);

  const VariantMeasurement eager_measurement = measure_variant(
      "eager", eager_factory, demo_streams.stream_index_to_stream);
  const VariantMeasurement compiled_measurement = measure_variant(
      "compiled", compiled_factory, demo_streams.stream_index_to_stream);

  print_variant_console(eager_measurement);
  print_variant_console(compiled_measurement);

  std::printf("\n");
  const ReportPaths eager_paths =
      write_variant_report(eager_measurement, "hello_world_eager");
  const ReportPaths compiled_paths =
      write_variant_report(compiled_measurement, "hello_world_compiled");
  write_comparison_report(eager_measurement, eager_paths, compiled_measurement,
                          compiled_paths);
  return 0;
}
