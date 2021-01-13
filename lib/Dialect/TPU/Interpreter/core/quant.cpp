#include "tpuc/Interpreter/cpu/quant.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {
QuantOpKernel::QuantOpKernel(Operation &op, value_map_t &valueMapping) {
  auto quantOp = cast<tpu::QuantOp>(op);
  assert(quantOp);
  llvm::outs() << " Quant op: [" << quantOp.name() << "]\n";

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = quantOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  llvm::outs() << "    =>required memory size: [" << size << "]\n";
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  this->name = quantOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  this->scale = quantOp.scale().convertToFloat();
  this->zero_point = quantOp.zero_point();
  this->from = quantOp.from().str();
  this->to = quantOp.to().str();

  set_datatype(getOpQuant(&op).str());
  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void QuantOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " Quant op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> QuantOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void QuantOpKernel::invoke() {
  for (size_t i = 0; i < input_data->size(); ++i) {
    output_data->at(i) = input_data->at(i) > 0 ? input_data->at(i) : 0;
  }
}

void QuantOpKernel::dump() {

  OpKernel::dump();
  llvm::outs() << "\tScale: " << this->scale << "\n";
  llvm::outs() << "\tFrom: " << this->from << "\n";
  llvm::outs() << "\tTo: " << this->to << "\n";
}
} // namespace mlir