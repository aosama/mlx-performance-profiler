#include "node_work.h"

namespace mlx_perf {

void add_node_work(NodeWork& total, const NodeWork& addend) {
  total.bytes_read += addend.bytes_read;
  total.bytes_written += addend.bytes_written;
  if (addend.is_arithmetic_counted) {
    total.is_arithmetic_counted = true;
    total.arithmetic_operations += addend.arithmetic_operations;
  }
}

} // namespace mlx_perf
