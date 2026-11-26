#include "demo_model.h"

#include "mlx/compile.h"
#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/transforms.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace mlx_perf {
namespace {

using mlx::core::array;
using mlx::core::Stream;

// Sized so the GPU does measurable work (a 256×1024×1024 addmm and a 2M
// element elementwise chain) while the whole job stays a few milliseconds.
constexpr int BATCH_ROWS = 256;
constexpr int FEATURE_WIDTH = 1024;
constexpr int REDUCTION_ROWS = 2048;
constexpr std::size_t DEMO_INPUT_COUNT = 4;

std::unordered_map<int, Stream> index_streams(
    const std::vector<Stream>& demo_stream_list) {
  std::unordered_map<int, Stream> stream_index_to_stream;
  // emplace: Stream has no default constructor, so operator[] cannot work.
  for (const Stream& demo_stream : demo_stream_list) {
    stream_index_to_stream.emplace(demo_stream.index, demo_stream);
  }
  return stream_index_to_stream;
}

} // namespace

DemoStreams make_demo_streams() {
  const Stream main_stream =
      mlx::core::default_stream(mlx::core::default_device());
  const Stream first_branch_stream =
      mlx::core::new_stream(mlx::core::default_device());
  const Stream second_branch_stream =
      mlx::core::new_stream(mlx::core::default_device());
  return {main_stream, first_branch_stream, second_branch_stream,
          index_streams({main_stream, first_branch_stream, second_branch_stream})};
}

std::vector<array> make_materialized_demo_inputs(const DemoStreams& demo_streams) {
  const Stream& input_stream = demo_streams.main_stream;
  std::vector<array> demo_inputs = {
      mlx::core::random::normal({BATCH_ROWS, FEATURE_WIDTH}, mlx::core::float32,
                                std::nullopt, input_stream),
      mlx::core::random::normal({BATCH_ROWS, FEATURE_WIDTH}, mlx::core::float32,
                                std::nullopt, input_stream),
      mlx::core::random::normal({FEATURE_WIDTH, FEATURE_WIDTH}, mlx::core::float32,
                                std::nullopt, input_stream),
      mlx::core::random::normal({REDUCTION_ROWS, FEATURE_WIDTH}, mlx::core::float32,
                                std::nullopt, input_stream)};
  mlx::core::eval(demo_inputs);
  return demo_inputs;
}

DemoComputation::DemoComputation(const DemoStreams& demo_streams)
    : main_stream_(demo_streams.main_stream),
      first_branch_stream_(demo_streams.first_branch_stream),
      second_branch_stream_(demo_streams.second_branch_stream) {}

std::vector<array> DemoComputation::operator()(
    const std::vector<array>& demo_inputs) const {
  using namespace mlx::core;
  if (demo_inputs.size() != DEMO_INPUT_COUNT) {
    throw std::invalid_argument(
        "DemoComputation expects " + std::to_string(DEMO_INPUT_COUNT) +
        " inputs, got " + std::to_string(demo_inputs.size()));
  }
  const array& shared_input = demo_inputs[0];
  const array& first_branch_input = demo_inputs[1];
  const array& first_branch_weights = demo_inputs[2];
  const array& second_branch_input = demo_inputs[3];

  // Branch one (stream 1): a dense layer with a SiLU activation — the
  // sigmoid·multiply tail is what compile can fuse.
  const array projected = addmm(shared_input, first_branch_input,
                                first_branch_weights, 1.f, 1.f, first_branch_stream_);
  const array first_branch = multiply(
      projected, sigmoid(projected, first_branch_stream_), first_branch_stream_);

  // Branch two (stream 2): an elementwise chain feeding a column reduction,
  // independent of branch one so the streams can overlap.
  const array half = array(0.5f);
  const array mixed = add(
      multiply(exp(second_branch_input, second_branch_stream_), half,
               second_branch_stream_),
      sin(second_branch_input, second_branch_stream_), second_branch_stream_);
  const array second_branch = multiply(
      arange(FEATURE_WIDTH, float32, second_branch_stream_),
      sum(mixed, 0, false, second_branch_stream_), second_branch_stream_);

  // The join crosses back onto the main stream: the Spark-shuffle edge.
  const array joined = add(first_branch, second_branch, main_stream_);
  const array activated =
      multiply(tanh(joined, main_stream_), array(2.0f), main_stream_);
  return {softmax(activated, std::vector<int>{-1}, false, main_stream_)};
}

EagerDemoGraphFactory::EagerDemoGraphFactory(
    const DemoStreams& demo_streams, std::vector<array> demo_inputs)
    : demo_computation_(demo_streams), demo_inputs_(std::move(demo_inputs)) {}

std::vector<array> EagerDemoGraphFactory::build_fresh_graph() const {
  return demo_computation_(demo_inputs_);
}

CompiledDemoGraphFactory::CompiledDemoGraphFactory(
    const DemoStreams& demo_streams, std::vector<array> demo_inputs)
    : compiled_computation_(mlx::core::compile(DemoComputation(demo_streams))),
      demo_inputs_(std::move(demo_inputs)) {}

std::vector<array> CompiledDemoGraphFactory::build_fresh_graph() const {
  return compiled_computation_(demo_inputs_);
}

} // namespace mlx_perf
