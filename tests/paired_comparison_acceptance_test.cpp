// Acceptance test for #12 on the Metal GPU: two graphs of one quantized
// projection, differing only in the primitive that computes it, produce one
// page stating which is faster, by how much, and whether the difference
// exceeds the spread. Spans paired_measurement, report_bundle and the
// verdict section, so it is named for the behaviour, not one module.

#include "json_value.h"
#include "paired_measurement.h"
#include "report_bundle.h"
#include "test_harness.h"

#include "mlx/mlx.h"

#include <cmath>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using namespace mlx::core;
using mlx_perf::ComparedJob;
using mlx_perf::GraphFactory;
using mlx_perf::PairedMeasurement;

constexpr int PROJECTION_WIDTH = 2048;

// One activation row and a 4-bit weight matrix, materialized once so every
// fresh graph starts from the same data (only the projection is measured).
struct QuantizedProjectionInputs {
  array activations = random::normal({1, PROJECTION_WIDTH});
  std::vector<array> quantized_weights =
      quantize(random::normal({PROJECTION_WIDTH, PROJECTION_WIDTH}));
  QuantizedProjectionInputs() {
    std::vector<array> materialized_inputs = quantized_weights;
    materialized_inputs.push_back(activations);
    eval(materialized_inputs);
  }
};

// The fused kernel: reads the packed weights directly.
class FusedProjectionFactory final : public GraphFactory {
 public:
  explicit FusedProjectionFactory(const QuantizedProjectionInputs& inputs) : inputs_(inputs) {}
  std::vector<array> build_fresh_graph() const override {
    return {quantized_matmul(inputs_.activations, inputs_.quantized_weights[0],
                             inputs_.quantized_weights[1], inputs_.quantized_weights[2])};
  }

 private:
  const QuantizedProjectionInputs& inputs_;
};

// The same projection through two primitives: dequantize to a full float
// matrix, then an ordinary matmul.
class DequantizeThenMatmulFactory final : public GraphFactory {
 public:
  explicit DequantizeThenMatmulFactory(const QuantizedProjectionInputs& inputs)
      : inputs_(inputs) {}
  std::vector<array> build_fresh_graph() const override {
    const array dequantized_weights =
        dequantize(inputs_.quantized_weights[0], inputs_.quantized_weights[1],
                   inputs_.quantized_weights[2]);
    return {matmul(inputs_.activations, transpose(dequantized_weights))};
  }

 private:
  const QuantizedProjectionInputs& inputs_;
};

std::string read_file(const std::string& file_path) {
  std::ifstream input_stream(file_path);
  return std::string(std::istreambuf_iterator<char>(input_stream), {});
}

bool contains(const std::string& text, const std::string& needle) {
  return text.find(needle) != std::string::npos;
}

void should_state_which_quantized_projection_is_faster_by_how_much_and_against_the_spread() {
  const QuantizedProjectionInputs inputs;
  const DequantizeThenMatmulFactory dequantize_then_matmul(inputs);
  const FusedProjectionFactory fused_projection(inputs);
  const Stream job_stream = default_stream(Device::gpu);
  const std::unordered_map<int, Stream> stream_index_to_stream{{job_stream.index, job_stream}};
  const PairedMeasurement paired_measurement = mlx_perf::measure_pair(
      ComparedJob{"dequantize_matmul", dequantize_then_matmul},
      ComparedJob{"quantized_matmul", fused_projection}, stream_index_to_stream);
  EXPECT_TRUE(mlx_perf::write_report_bundle(
      {"Quantized projection 2048×2048, 4-bit", ".", "acceptance_quantized_projection"},
      paired_measurement));

  // One page, opening with the verdict: which, by how much, against the spread.
  const std::string comparison_page = read_file("acceptance_quantized_projection.html");
  const std::size_t verdict_at = comparison_page.find("<p class=\"verdict");
  EXPECT_TRUE(verdict_at != std::string::npos);
  const std::string verdict_sentence =
      comparison_page.substr(verdict_at, comparison_page.find("</p>", verdict_at) - verdict_at);
  EXPECT_TRUE(contains(verdict_sentence, "quantized_matmul"));
  EXPECT_TRUE(contains(verdict_sentence, "interval for the per-round difference"));
  EXPECT_TRUE(contains(verdict_sentence, "% faster than") ||
              contains(verdict_sentence, "% slower than") ||
              contains(verdict_sentence, "is not a result"));

  // The dequantizing path writes and re-reads a full float matrix that the
  // fused kernel never materializes: over the whole job's GPU span (one
  // timed unit per round, so node count adds no instrumentation) the fused
  // kernel must win beyond the spread. Timing-dependent by nature; the
  // margin is about 6x.
  const mlx_perf::JsonValue comparison_record =
      mlx_perf::parse_json_file("acceptance_quantized_projection.json");
  EXPECT_TRUE(comparison_record.at("metrics").at("gpu_span_us").at("verdict").as_string() ==
              "candidate_faster");
  // Same projection, two primitives: the outputs agree closely.
  EXPECT_TRUE(paired_measurement.output_agreement.output_count == 1);
  EXPECT_TRUE(std::isfinite(paired_measurement.output_agreement.max_absolute_difference));
  EXPECT_TRUE(paired_measurement.output_agreement.max_absolute_difference < 1.0);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(
          should_state_which_quantized_projection_is_faster_by_how_much_and_against_the_spread),
  });
}
