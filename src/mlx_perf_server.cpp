// mlx_perf_server: the live Spark-UI-style view. Orchestration only.
//
// Threading model: this (main) thread owns every MLX call — measuring,
// rendering, memory sampling — because MLX's default streams and Metal
// encoders are per-thread. HTTP threads only read the LiveViewStore.

#include "demo_model.h"
#include "http_server.h"
#include "live_view_renderer.h"
#include "live_view_routes.h"
#include "live_view_store.h"
#include "measurement_session.h"
#include "paired_measurement.h"
#include "report_verdict_section.h"
#include "server_options.h"

#include "mlx/memory.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace mlx_perf;
using Clock = std::chrono::steady_clock;

const std::string JOB_TITLE = "Three-stream toy job";
constexpr auto MEMORY_SAMPLE_INTERVAL = std::chrono::milliseconds(1000);

volatile std::sig_atomic_t stop_signal_received = 0;

void handle_stop_signal(int) { stop_signal_received = 1; }

class StoreStatusObserver final : public MeasurementObserver {
 public:
  explicit StoreStatusObserver(LiveViewStore& live_view_store)
      : live_view_store_(live_view_store) {}
  void on_pass_started(const std::string& variant_name,
                       const std::string& pass_description) override {
    live_view_store_.set_status("measuring " + variant_name + ": " + pass_description);
  }

 private:
  LiveViewStore& live_view_store_;
};

// Eager against compiled as an interleaved pair, the same way the static
// comparison page measures them, so the live Job tab opens with the verdict.
void measure_and_publish(LiveViewStore& live_view_store, const DemoStreams& demo_streams,
                         const GraphFactory& eager_factory,
                         const GraphFactory& compiled_factory) {
  StoreStatusObserver status_observer(live_view_store);
  const PairedMeasurement paired_measurement =
      measure_pair({"eager", eager_factory}, {"compiled", compiled_factory},
                   demo_streams.stream_index_to_stream, {}, &status_observer);
  live_view_store.publish_views(render_live_views(JOB_TITLE, paired_measurement));
  live_view_store.set_status(
      "idle — " +
      describe_verdict("eager", "compiled", compare_pair(paired_measurement).job_wall) +
      " Re-run to measure again.");
}

void sample_memory(LiveViewStore& live_view_store, Clock::time_point server_started_at) {
  MemorySample memory_sample;
  memory_sample.seconds_since_start =
      std::chrono::duration<double>(Clock::now() - server_started_at).count();
  memory_sample.active_bytes = mlx::core::get_active_memory();
  memory_sample.peak_bytes = mlx::core::get_peak_memory();
  memory_sample.cache_bytes = mlx::core::get_cache_memory();
  live_view_store.add_memory_sample(memory_sample);
}

} // namespace

int main(int argument_count, char** argument_values) {
  const ServerOptions server_options = parse_server_options(
      std::vector<std::string>(argument_values + 1, argument_values + argument_count));
  if (server_options.should_show_help || !server_options.error_message.empty()) {
    if (!server_options.error_message.empty()) {
      std::fprintf(stderr, "error: %s\n", server_options.error_message.c_str());
    }
    std::fputs(server_usage_text().c_str(), stderr);
    return server_options.error_message.empty() ? 0 : 2;
  }
  LiveViewStore live_view_store;
  LiveViewRoutes live_view_routes(live_view_store, JOB_TITLE);
  HttpServer http_server(live_view_routes);
  const std::uint16_t bound_port = http_server.start(server_options.port);
  if (bound_port == 0) {
    std::fprintf(stderr, "error: cannot listen on 127.0.0.1:%u\n", server_options.port);
    return 1;
  }
  std::signal(SIGINT, handle_stop_signal);
  std::signal(SIGTERM, handle_stop_signal);
  std::printf("live view: http://127.0.0.1:%u/  (Ctrl-C to stop)\n", bound_port);

  const Clock::time_point server_started_at = Clock::now();
  const DemoStreams demo_streams = make_demo_streams();
  const auto demo_inputs = make_materialized_demo_inputs(demo_streams);
  const EagerDemoGraphFactory eager_factory(demo_streams, demo_inputs);
  const CompiledDemoGraphFactory compiled_factory(demo_streams, demo_inputs);
  measure_and_publish(live_view_store, demo_streams, eager_factory, compiled_factory);
  while (stop_signal_received == 0) {
    if (live_view_store.wait_for_run_request(MEMORY_SAMPLE_INTERVAL)) {
      measure_and_publish(live_view_store, demo_streams, eager_factory, compiled_factory);
    }
    sample_memory(live_view_store, server_started_at);
  }
  std::printf("\nstopping...\n");
  live_view_routes.close_event_streams();
  http_server.stop();
  return 0;
}
