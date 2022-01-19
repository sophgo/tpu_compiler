#include "tpuc/Interpreter/cpu/copy.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/MlirModuleInterpreter.h"

namespace mlir {

CopyOpKernel::CopyOpKernel(Operation &op, value_map_t &valueMapping,
                           weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
  parseCopyParam<tpu::CopyOp>(&op, copy_shape, input_stride, output_stride);
}

void CopyOpKernel::invoke() {
  for(int n = 0; n < copy_shape[0]; n++) {
    for (int c = 0; c < copy_shape[1]; c++) {
      for (int h = 0; h < copy_shape[2]; h++) {
        for (int w = 0; w < copy_shape[3]; w++) {
          int in_index = n * input_stride[0] + c * input_stride[1] + h * input_stride[2] + w * input_stride[3];
          int out_index = n * output_stride[0] + c * output_stride[1] + h * output_stride[2] + w * output_stride[3];
          output_data->at(out_index) = input_data->at(in_index);
        }
      }
    }
  }
}

} // namespace mlir