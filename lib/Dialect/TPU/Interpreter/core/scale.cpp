#include "tpuc/Interpreter/cpu/scale.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {
ScaleOpKernel::ScaleOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  input_data = this->opdTensors[0];
  scale = this->opdTensors[1];
  bias = this->opdTensors[2];
  output_data = this->resTensor;
}

void ScaleOpKernel::invoke() {
  if (datatype != DataType::FP32) {
    llvm_unreachable("except fp32, other mode please fused batchnorm");
  }
  int n = this->shape.at(0);
  int c = this->shape.at(1);
  int h = this->shape.size() > 2 ? this->shape.at(2) : 1;
  int w = this->shape.size() > 3 ? this->shape.at(3) : 1;
  int planner = h * w;
#pragma omp parallel for collapse(3)
  for (int ni = 0; ni < n; ++ni) {
    for (int ci = 0; ci < c; ++ci) {
      for (int i = 0; i < planner; ++i) {
        auto x = input_data->at(ni * c * h * w + ci * h * w + i);
        auto y = x * scale->at(ci);
        if (bias) {
          y += bias->at(ci);
        }
        output_data->at(ni * c * h * w + ci * h * w + i) = y;
      }
    }
  }
}
void ScaleOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " Scale op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
}

std::vector<float> ScaleOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}
void ScaleOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir