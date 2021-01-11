#include "tpuc/Interpreter/cpu/activation.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {
ReluOpKernel::ReluOpKernel(Operation &op, value_map_t &valueMapping) {
  auto reluOp = cast<tpu::ReluOp>(op);
  assert(reluOp);
  llvm::outs() << " Relu op: [" << reluOp.name() << "]\n";

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = reluOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);

  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  this->name = reluOp.name().str();
  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void ReluOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " Relu op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> ReluOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void ReluOpKernel::invoke() {
  for (size_t i = 0; i < input_data->size(); ++i) {
    output_data->at(i) = input_data->at(i) > 0 ? input_data->at(i) : 0;
  }
}

void ReluOpKernel::dump() {
  std::string shape_str;
  for (auto &i : this->shape) {
    shape_str = shape_str + std::to_string(i) + " ";
  }

  llvm::outs() << "Relu Op\n";
  llvm::outs() << "\tName: " << this->name << "\n";
  llvm::outs() << "\tShape: " << shape_str << "\n";
}
} // namespace mlir