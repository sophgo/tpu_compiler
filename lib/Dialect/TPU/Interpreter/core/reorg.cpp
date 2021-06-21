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
  int out_c = input_shape[1] * stride * stride;
  int out_h = input_shape[2] / stride;
  int out_w = input_shape[3] / stride;
  int in_c = input_shape[1];
  int in_h = input_shape[2];
  int in_w = input_shape[3];
  for (int b = 0; b < n; b++) {
    for (int k = 0; k < out_c; k++) {
      for (int j = 0; j < out_h; j++) {
        for (int i = 0; i < out_w; i++) {
          int in_index = i + out_w * (j + out_h * (k + out_c * b));
          int c2 = k % in_c;
          int offset = k / in_c;
          int w2 = i * stride + offset % stride;
          int h2 = j * stride + offset / stride;
          int out_index = w2 + in_w * (h2 + in_h * (c2 + in_c * b));
          output_data->at(in_index) = input_data->at(out_index);
        }
      }
    }
  }
}

} // namespace mlir