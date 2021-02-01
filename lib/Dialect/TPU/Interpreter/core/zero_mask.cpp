#include "tpuc/Interpreter/cpu/zero_mask.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {

ZeroMaskOpKernel::ZeroMaskOpKernel(Operation &op, value_map_t &valueMapping) {
  auto zero_maskOp = cast<tpu::ZeroMaskOp>(op);
  assert(zero_maskOp);
  LLVM_DEBUG(llvm::outs() << " ZeroMaskOp op: [" << zero_maskOp.name()
                          << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = zero_maskOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  auto input_type = zero_maskOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();

  this->name = zero_maskOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());

  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void ZeroMaskOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " ZeroMaskOp op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> ZeroMaskOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void ZeroMaskOpKernel::invoke() {
  for (uint32_t i = 0; i < output_data->size(); i++) {
    output_data->at(i) = (input_data->at(i) == 0.0 ? 1.0 : 0.0);
  }
}
void ZeroMaskOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir