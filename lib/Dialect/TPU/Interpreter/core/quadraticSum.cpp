#include "tpuc/Interpreter/cpu/quadraticSum.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/conv.hpp"
#include "tpuc/MlirModuleInterpreter.h"
#include "tpuc/QuantizationArithmetic.h"

namespace mlir {

QuadraticSumOpKernel::QuadraticSumOpKernel(Operation &op,
                                           value_map_t &valueMapping,
                                           weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {

  this->input_shape = getTensorShape(op.getOperand(0));
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void QuadraticSumOpKernel::invoke() {
  int n = input_shape[0];
  int c = input_shape[1];
  int h = input_shape[2];
  int w = input_shape[3];
  mkldnn_conv(input_data->data(), input_data->data(), nullptr,
              output_data->data(), n, c, h, w, c, 1, 1, h, w, h, w, 1, 1,
              0, 0, 0, 0, c, 0);
  if (datatype == DataType::BF16) {
    BF16(output_data->data(), output_data->data(), output_data->size());
  }
}

} // namespace mlir