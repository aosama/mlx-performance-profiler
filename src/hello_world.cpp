// The demo: measure one small three-stream job twice — as written (eager) and
// through mlx::core::compile — and write a Spark-style report for each plus
// a side-by-side comparison. Orchestration only: the job lives in
// demo_model, the passes in measurement_session, rendering in report_*.

#include "demo_model.h"
#include "measurement_session.h"
#include "report_bundle.h"
#include "timeline_console.h"

#include <cstdio>

namespace {

using namespace mlx_plan;

void print_variant_console(const VariantMeasurement& variant_measurement) {
  std::printf("\n######## %s ########", variant_measurement.variant_name.c_str());
  print_job_summary(variant_measurement.job_totals, variant_measurement.gpu_timeline);
  print_stream_table(variant_measurement.gpu_timeline);
  print_node_table(variant_measurement.gpu_timeline);
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
  const bool are_reports_written = write_report_bundle(
      {"Hello World — three-stream toy job", "build", "hello_world"}, eager_measurement,
      compiled_measurement);
  return are_reports_written ? 0 : 1;
}
