#include "tpuc/Interpreter/cpu/scale.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {
ScaleOpKernel::ScaleOpKernel(Operation &op, value_map_t &valueMapping) {
  auto scaleOp = cast<tpu::ScaleOp>(op);
  assert(scaleOp);
  llvm::outs() << " Scale op: [" << scaleOp.name() << "]\n";

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = scaleOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);

  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  this->name = scaleOp.name().str();
  // get tensors
  input_data = opTensors[0];
  scale = opTensors[1];
  bias = opTensors[2];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void ScaleOpKernel::invoke() {
  int n = this->shape.at(0);
  int c = this->shape.at(1);
  int h = this->shape.at(2);
  int w = this->shape.at(3);
  for (int ni = 0; ni < n; ++ni) {
    for (int ci = 0; ci < c; ++ci) {
      for (int i = 0; i < h * w; ++i) {
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
};
std::vector<float> ScaleOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}
void ScaleOpKernel::dump() {
  std::string shape_str;
  for (auto &i : this->shape) {
    shape_str = shape_str + std::to_string(i) + " ";
  }

  llvm::outs() << "Scale Op\n";
  llvm::outs() << "\tName: " << this->name << "\n";
  llvm::outs() << "\tShape: " << shape_str << "\n";
}
} // namespace mlir