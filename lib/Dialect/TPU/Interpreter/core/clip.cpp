#include "tpuc/Interpreter/cpu/clip.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {

ClipOpKernel::ClipOpKernel(Operation &op, value_map_t &valueMapping) {
  auto clipOp = cast<tpu::ClipOp>(op);
  assert(clipOp);
  llvm::outs() << " ClipOp op: [" << clipOp.name() << "]\n";

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = clipOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  llvm::outs() << "    =>required memory size: [" << size << "]\n";
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  auto input_type = clipOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();

  this->name = clipOp.name().str();
  this->max = clipOp.max().convertToFloat();
  this->min = clipOp.min().convertToFloat();

  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());

  if (datatype == DataType::INT8) {
    auto quant_rshift = opTensors[3];
    auto quant_multiplier = opTensors[4];

    this->rshift = quant_rshift->at(0);
    this->multiplier = quant_multiplier->at(0);
  }
  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void ClipOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " ClipOp op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> ClipOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void ClipOpKernel::invoke() {
  for (size_t i = 0; i < output_data->size(); i++) {
    output_data->at(i) = std::max(min, std::min(input_data->at(i), max));
  }
  if (datatype == DataType::FP32) {

  } else if (datatype == DataType::INT8) {
    for (size_t i = 0; i < output_data->size(); ++i) {
      output_data->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
          output_data->at(i), (uint32_t)rshift, (uint32_t)multiplier, false);
    }
  } else {
    clean16bitmantissa(output_data->data(), output_data->data(),
                       output_data->size());
  }
}

void ClipOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir