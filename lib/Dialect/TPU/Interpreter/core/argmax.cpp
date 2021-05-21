#include "tpuc/Interpreter/cpu/argmax.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"
#include <limits>

namespace mlir {

ArgMaxOpKernel::ArgMaxOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto argmaxOp = cast<tpu::ArgMaxOp>(op);
  auto input_type = argmaxOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();
  this->axis = argmaxOp.axis();
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void ArgMaxOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " ArgMaxOp op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> ArgMaxOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void ArgMaxOpKernel::invoke() {
  float *input = input_data->data();
  float *output = output_data->data();
  int w = input_shape[input_shape.size() - 1];
  int total = std::accumulate(
      input_shape.begin(), input_shape.end(),
      1, std::multiplies<>());
  for (int i = 0; i < total/w; i++) {
    float max_value = -999999999.0;
    int idx = -1;
    float *ptr = input + i * w;
    for (int j = 0; j < w; j++) {
      if (ptr[j] > max_value) {
        idx = j;
        max_value = ptr[j];
      }
    }
    output[i] = idx;
  }
}

void ArgMaxOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir