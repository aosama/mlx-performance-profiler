#include "machine_ceilings.h"

#include "console_output.h"
#include "graph_factory.h"
#include "graph_model.h"
#include "stage_timeline.h"

#include "mlx/memory.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"

#include <limits>
#include <vector>

namespace mlx_perf {
namespace {

using mlx::core::array;

// Far larger than any system level cache Apple ships, so the reduction reads
// unified memory itself and not a cache: what a model's weights cost.
constexpr int READING_CALIBRATION_ELEMENT_COUNT = 128 * 1024 * 1024;  // float32: 512 MiB
// Large enough that MLX picks its fastest multiply and launch costs vanish
// against about 137 billion operations.
constexpr int ARITHMETIC_CALIBRATION_SIDE = 4096;
// The best of several: a ceiling is what the GPU can do, so the fastest
// round is the truest one; slower rounds measured interference.
constexpr int CALIBRATION_ROUND_COUNT = 5;

// Reading: one reduction over the whole buffer. Its arithmetic (one add per
// element) is far below what the GPU can do, so the read decides its time.
class FullReductionFactory : public GraphFactory {
 public:
  explicit FullReductionFactory(array source_values) : source_values_(std::move(source_values)) {}
  std::vector<array> build_fresh_graph() const override {
    return {mlx::core::sum(source_values_)};
  }

 private:
  array source_values_;
};

class SquareMultiplyFactory : public GraphFactory {
 public:
  SquareMultiplyFactory(array left_matrix, array right_matrix)
      : left_matrix_(std::move(left_matrix)), right_matrix_(std::move(right_matrix)) {}
  std::vector<array> build_fresh_graph() const override {
    return {mlx::core::matmul(left_matrix_, right_matrix_)};
  }

 private:
  array left_matrix_;
  array right_matrix_;
};

// The shortest GPU span over the rounds, in seconds; 0 when the GPU gave no
// window. One warm-up first absorbs kernel compilation.
double measure_shortest_gpu_seconds(const GraphFactory& graph_factory) {
  mlx::core::eval(graph_factory.build_fresh_graph());
  double shortest_microseconds = std::numeric_limits<double>::max();
  for (int round_index = 0; round_index < CALIBRATION_ROUND_COUNT; ++round_index) {
    const CapturedGraph calibration_graph = capture_graph(graph_factory.build_fresh_graph());
    const GpuTimeline calibration_timeline = capture_stage_timeline(
        calibration_graph, collect_streams(calibration_graph), {}, "calibration");
    const double span_microseconds = calibration_timeline.measured_gpu_compute_microseconds;
    if (span_microseconds > 0.0 && span_microseconds < shortest_microseconds) {
      shortest_microseconds = span_microseconds;
    }
  }
  return shortest_microseconds == std::numeric_limits<double>::max()
      ? 0.0 : shortest_microseconds * 1e-6;
}

double measure_reading_bytes_per_second() {
  array source_values = mlx::core::ones({READING_CALIBRATION_ELEMENT_COUNT}, mlx::core::float32);
  source_values.eval();
  const double shortest_seconds =
      measure_shortest_gpu_seconds(FullReductionFactory(source_values));
  return shortest_seconds > 0.0 ? static_cast<double>(source_values.nbytes()) / shortest_seconds : 0.0;
}

double measure_multiply_operations_per_second(mlx::core::Dtype matrix_dtype) {
  array left_matrix = mlx::core::ones({ARITHMETIC_CALIBRATION_SIDE, ARITHMETIC_CALIBRATION_SIDE}, matrix_dtype);
  array right_matrix = mlx::core::ones({ARITHMETIC_CALIBRATION_SIDE, ARITHMETIC_CALIBRATION_SIDE}, matrix_dtype);
  mlx::core::eval({left_matrix, right_matrix});
  const double shortest_seconds =
      measure_shortest_gpu_seconds(SquareMultiplyFactory(left_matrix, right_matrix));
  const double side = ARITHMETIC_CALIBRATION_SIDE;
  return shortest_seconds > 0.0 ? 2.0 * side * side * side / shortest_seconds : 0.0;
}

// Models run in either half-precision type and MLX's kernels differ in speed
// between them, so the faster of the two is the ceiling.
void measure_arithmetic_ceiling(MachineCeilings& machine_ceilings) {
  const double float16_rate = measure_multiply_operations_per_second(mlx::core::float16);
  const double bfloat16_rate = measure_multiply_operations_per_second(mlx::core::bfloat16);
  const bool is_bfloat16_faster = bfloat16_rate > float16_rate;
  machine_ceilings.arithmetic_operations_per_second = is_bfloat16_faster ? bfloat16_rate : float16_rate;
  machine_ceilings.arithmetic_dtype = is_bfloat16_faster ? "bfloat16" : "float16";
}

MachineCeilings calibrate_machine_ceilings() {
  print_progress_line("[ceilings] measuring this machine's reading and arithmetic ceilings...\n");
  MachineCeilings machine_ceilings;
  machine_ceilings.memory_read_bytes_per_second = measure_reading_bytes_per_second();
  measure_arithmetic_ceiling(machine_ceilings);
  machine_ceilings.is_measured = machine_ceilings.memory_read_bytes_per_second > 0.0 &&
      machine_ceilings.arithmetic_operations_per_second > 0.0;
  // The calibration buffers return to MLX's pool when freed; clearing it
  // keeps them out of every memory reading taken after this point.
  mlx::core::clear_cache();
  return machine_ceilings;
}

} // namespace

const MachineCeilings& read_machine_ceilings() {
  static const MachineCeilings machine_ceilings = calibrate_machine_ceilings();
  return machine_ceilings;
}

} // namespace mlx_perf
