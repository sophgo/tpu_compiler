#include "tpuc/Interpreter/cpu/reflectionpad.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {

ReflectionPadOpKernel::ReflectionPadOpKernel(Operation &op,
                                                 value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto reflectionpadOp = cast<tpu::ReflectionPadOp>(op);
  auto input_type =
      reflectionpadOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();
  arrayAttrToVector(reflectionpadOp.pads(), this->pad);
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void ReflectionPadOpKernel::invoke() {
  int N = std::accumulate(input_shape.begin(), input_shape.end() - 1, 1,
                          std::multiplies<int64_t>());
  int K = input_shape.back();
  int OK = K + pad[0] + pad[1];

  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      int out_idx = n * OK + k + pad[0];
      int in_idx = n * K + k;
      output_data->at(out_idx) = input_data->at(in_idx);
    }
  }

  for (int n = 0; n < N; ++n) {
    for (int p = 0; p < pad[0]; ++p) {
      int in_idx = n * K + p + 1;
      int out_idx = n * OK + pad[0] - p - 1;
      output_data->at(out_idx) = input_data->at(in_idx);
    }
    for (int p = 0; p < pad[1]; ++p) {
      int in_idx = n * K + K - p - 2;
      int out_idx = n * OK + OK - pad[1] + p;
      output_data->at(out_idx) = input_data->at(in_idx);
    }
  }
}

void ReflectionPadOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir
