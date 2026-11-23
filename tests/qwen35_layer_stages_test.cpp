// Integration tests (Metal GPU) for qwen35_layer_stages: the stage cuts the
// Qwen3.5 decode job hands the generic profiler, on the tiny synthetic model.

#include "graph_model.h"
#include "qwen35/qwen35_decode_job.h"
#include "qwen35/qwen35_layer_stages.h"
#include "stage_timeline.h"
#include "test_harness.h"
#include "tiny_qwen35_model.h"

#include "mlx/mlx.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace tiny_qwen35_model;
using mlx_plan::GpuTimeline;
using mlx_plan::StagedGraph;

GpuTimeline time_staged_graph(const StagedGraph& staged_graph) {
  const mlx_plan::CapturedGraph job_graph = mlx_plan::capture_graph(staged_graph.output_arrays);
  return mlx_plan::capture_stage_timeline(
      job_graph, mlx_plan::collect_streams(job_graph),
      mlx_plan::resolve_stage_cuts(job_graph, staged_graph.stage_boundaries),
      staged_graph.final_stage_name);
}

std::vector<std::string> distinct_stage_names(const GpuTimeline& gpu_timeline) {
  std::vector<std::string> stage_names;
  for (const auto& stage_task : gpu_timeline.stages) {
    if (stage_names.empty() || stage_names.back() != stage_task.stage_name) {
      stage_names.push_back(stage_task.stage_name);
    }
  }
  return stage_names;
}

// --- positive ---

// Eager: one stage per decoder layer, then the LM head — and every stage
// claims real work (none dropped as empty).
void should_time_one_stage_per_layer_for_the_eager_step() {
  const Qwen35TextConfig config = make_tiny_config();
  const Qwen35Weights weights = make_tiny_weights(config);
  const EagerDecodeGraphFactory eager_factory(weights, config, make_profiling_inputs(config, 5, 9));
  const StagedGraph staged_graph = eager_factory.build_fresh_staged_graph();
  EXPECT_TRUE(staged_graph.stage_boundaries.size() == config.layer_kinds.size());
  EXPECT_TRUE(distinct_stage_names(time_staged_graph(staged_graph)) ==
              (std::vector<std::string>{"embed + layer 0", "layer 1", FINAL_STAGE_NAME}));
}

void should_cut_the_compiled_step_at_each_layer_state() {
  const Qwen35TextConfig config = make_tiny_config();
  const Qwen35Weights weights = make_tiny_weights(config);
  const CompiledDecodeGraphFactory compiled_factory(weights, config,
                                                    make_profiling_inputs(config, 5, 9));
  compiled_factory.build_fresh_graph();  // trace once, as the warm-up pass does
  const StagedGraph staged_graph = compiled_factory.build_fresh_staged_graph();
  EXPECT_TRUE(distinct_stage_names(time_staged_graph(staged_graph)) ==
              (std::vector<std::string>{"up to layer 0 state", "up to layer 1 state",
                                        FINAL_STAGE_NAME}));
}

// --- edge / boundary ---

// Delta-net layers contribute their conv window and recurrent state;
// attention layers their keys and values; both add the hidden state.
void should_collect_the_hidden_state_and_present_caches_per_layer() {
  LayerStageCollector layer_stage_collector;
  LayerCache delta_cache;
  delta_cache.conv_window = ones({1});
  delta_cache.delta_state = ones({1});
  layer_stage_collector.on_layer_built(0, ones({1}), delta_cache);
  layer_stage_collector.on_layer_built(1, ones({1}), LayerCache{});
  const auto stage_boundaries = layer_stage_collector.take_stage_boundaries();
  EXPECT_TRUE(stage_boundaries.size() == 2);
  if (stage_boundaries.size() == 2) {
    EXPECT_TRUE(stage_boundaries[0].boundary_arrays.size() == 3);
    EXPECT_TRUE(stage_boundaries[1].boundary_arrays.size() == 1);
    EXPECT_TRUE(stage_boundaries[1].stage_name == "layer 1");
  }
  EXPECT_TRUE(layer_stage_collector.take_stage_boundaries().empty());
}

// --- negative ---

void should_reject_compiled_outputs_that_do_not_match_the_layer_count() {
  const Qwen35TextConfig config = make_tiny_config();
  bool has_thrown = false;
  try {
    layer_state_boundaries({ones({1}), ones({1}), ones({1})}, config);
  } catch (const std::invalid_argument&) {
    has_thrown = true;
  }
  EXPECT_TRUE(has_thrown);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_time_one_stage_per_layer_for_the_eager_step),
      TEST_CASE(should_cut_the_compiled_step_at_each_layer_state),
      TEST_CASE(should_collect_the_hidden_state_and_present_caches_per_layer),
      TEST_CASE(should_reject_compiled_outputs_that_do_not_match_the_layer_count),
  });
}
