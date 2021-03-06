#include "tpuc/Interpreter/cpu/tile.hpp"
#include "internal.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/MlirModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"

namespace mlir {

TileOpKernel::TileOpKernel(Operation &op, value_map_t &valueMapping,
                           weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto tileOp = cast<tpu::TileOp>(op);
  auto shape_ = getTensorShape(tileOp.input());
  int64_t n, c, h, w;
  getNCHW(shape_, n, c, h, w);
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
  axis = tileOp.axis();
  tiles = tileOp.tiles();
  input_shape = {n, c, h, w};
}

void TileOpKernel::invoke() {

  int outer_count =
      std::accumulate(input_shape.begin(), input_shape.begin() + axis, 1,
                      std::multiplies<int>());
  int inner_count = std::accumulate(
      input_shape.begin() + axis, input_shape.end(), 1, std::multiplies<int>());
  auto input = input_data->data();
  auto output = output_data->data();
#pragma omp parallel for schedule(static, omp_schedule(outer_count))
  for (int out = 0; out < outer_count; ++out) {
    auto start = input + out * inner_count;
    auto end = start + inner_count;
    for (int t = 0; t < tiles; ++t) {
      std::copy(start, end,
                output + out * tiles * inner_count + t * inner_count);
    }
  }
}

} // namespace mlir