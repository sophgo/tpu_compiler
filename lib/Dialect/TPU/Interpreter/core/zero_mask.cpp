#include "tpuc/Interpreter/cpu/zero_mask.hpp"
#include "internal.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/lut_func.hpp"
#include "tpuc/MlirModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"

namespace mlir {

ZeroMaskOpKernel::ZeroMaskOpKernel(Operation &op, value_map_t &valueMapping,
                                   weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto cast_op = cast<tpu::ZeroMaskOp>(op);
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
  positive = cast_op.positive();
}

void ZeroMaskOpKernel::invoke() {
  auto total_num = std::accumulate(shape.begin(), shape.end(), 1,
                                   std::multiplies<int64_t>());
  auto *input = input_data->data();
  auto *output = output_data->data();
  float mask = (datatype == DataType::INT8 ? 127 : 1);
#pragma omp parallel for schedule(static, omp_schedule(total_num))
  for (int64_t i = 0; i < total_num; i++) {
    if (input[i] == 0) {
      output[i] = positive ? 0 : mask;
    } else {
      output[i] = positive ? mask : 0;
    }
  }
}

} // namespace mlir