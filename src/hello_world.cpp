// The demo: measure one small three-stream job two ways — as written (eager)
// and through mlx::core::compile — with their passes interleaved, and write a
// Spark-style report for each plus a side-by-side comparison with a verdict.
// Orchestration only: the job lives in demo_model, the passes in
// paired_measurement, rendering in report_*.

#include "demo_model.h"
#include "paired_measurement.h"
#include "report_bundle.h"
#include "timeline_console.h"
#include "dispatch_recorder.h"

#include <cstdio>

namespace {

using namespace mlx_perf;

void print_variant_console(const VariantMeasurement& variant_measurement) {
  std::printf("\n######## %s ########", variant_measurement.variant_name.c_str());
  print_job_summary(variant_measurement.job_totals, variant_measurement.gpu_timeline);
  print_stream_table(variant_measurement.gpu_timeline);
  print_node_table(variant_measurement.gpu_timeline);
}

} // namespace

int main() {
  // Before any MLX work, so every kernel launch the profile records is named.
  mlx_perf::install_dispatch_recorder();
  const DemoStreams demo_streams = make_demo_streams();
  std::printf("materializing demo inputs...\n");
  const auto demo_inputs = make_materialized_demo_inputs(demo_streams);
  const EagerDemoGraphFactory eager_factory(demo_streams, demo_inputs);
  const CompiledDemoGraphFactory compiled_factory(demo_streams, demo_inputs);

  const PairedMeasurement paired_measurement =
      measure_pair({"eager", eager_factory}, {"compiled", compiled_factory},
                   demo_streams.stream_index_to_stream);

  print_variant_console(paired_measurement.baseline);
  print_variant_console(paired_measurement.candidate);

  std::printf("\n");
  const bool are_reports_written = write_report_bundle(
      {"Hello World — three-stream toy job", "build", "hello_world"}, paired_measurement);
  return are_reports_written ? 0 : 1;
}
