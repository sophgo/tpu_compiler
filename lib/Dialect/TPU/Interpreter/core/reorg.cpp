#include "tpuc/Interpreter/cpu/reorg.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {

ReorgOpKernel::ReorgOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto reorgOp = cast<tpu::ReorgOp>(op);
  auto input_type = reorgOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();
  this->stride = reorgOp.stride();
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void ReorgOpKernel::invoke() {
  int n = input_shape[0];
  int c = input_shape[1];
  int h = input_shape[2];
  int w = input_shape[3];
  int out_c = c / (stride * stride);
  int out_w = w * stride;
  int out_h = h * stride;
  for (int b = 0; b < n; b++) {
    for (int k = 0; k < c; k++) {
      for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
          int in_index = i + w * (j + h * (k + c * b));
          int c2 = k % out_c;
          int offset = k / out_c;
          int w2 = i * stride + offset % stride;
          int h2 = j * stride + offset / stride;
          int out_index = w2 + out_w * (h2 + out_h * (c2 + out_c * b));
          output_data->at(in_index) = input_data->at(out_index);
        }
      }
    }
  }
}

} // namespace mlir