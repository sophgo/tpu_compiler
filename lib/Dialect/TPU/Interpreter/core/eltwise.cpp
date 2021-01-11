#include "tpuc/Interpreter/cpu/eltwise.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {
EltwiseAddOpKernel::EltwiseAddOpKernel(Operation &op,
                                       value_map_t &valueMapping) {

  auto elt_addOp = cast<tpu::EltwiseAddOp>(op);
  llvm::outs() << " Eltwise Add op: [" << elt_addOp.name() << "]\n";

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = elt_addOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);

  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  this->name = elt_addOp.name().str();

  // erase the end 4 elements:
  opTensors.erase(opTensors.end() - 4, opTensors.end());
  // get tensors
  inputs_data = opTensors;
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void EltwiseAddOpKernel::set_tensor(const std::vector<float> &data) {
  llvm_unreachable("TODO!");
};
std::vector<float> EltwiseAddOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}
void EltwiseAddOpKernel::invoke() {
  int in = this->shape.at(0);
  int ic = this->shape.at(1);
  int ih = this->shape.at(2);
  int iw = this->shape.at(3);
  std::fill(output_data->begin(), output_data->end(), 0);

  for (size_t ni = 0; ni < inputs_data.size(); ++ni) {
    for (size_t i = 0; i < (size_t)(in * ic * ih * iw); ++i) {
      output_data->at(i) += inputs_data[ni]->at(i);
    }
  }
};

void EltwiseAddOpKernel::dump() { llvm_unreachable("TODO!"); }
} // namespace mlir