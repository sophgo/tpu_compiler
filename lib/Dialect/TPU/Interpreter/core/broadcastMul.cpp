#include "tpuc/Interpreter/cpu/broadcastMul.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/activation.hpp"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {
BroadcastMulOpKernel::BroadcastMulOpKernel(Operation &op,
                                           value_map_t &valueMapping) {
  auto broadcastmulOp = cast<tpu::BroadcastMulOp>(op);
  assert(broadcastmulOp);
  LLVM_DEBUG(llvm::outs() << " BroadcastMul op: [" << broadcastmulOp.name()
                          << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = broadcastmulOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  do_relu = broadcastmulOp.do_relu();
  int axis = broadcastmulOp.axis();
  if (axis != 1) {
    llvm_unreachable("only support axis 1 broadcast mul");
  }

  this->name = broadcastmulOp.name().str();
  this->op_type = op.getName().getStringRef().str();

  set_datatype(getOpQuant(&op).str());
  if (datatype == DataType::INT8) {
    auto quant_rshift = opTensors[4];
    auto quant_multiplier = opTensors[5];
    assert(quant_rshift && quant_multiplier);
    rshift = quant_rshift->at(0);
    mutliplier = quant_multiplier->at(0);
  }
  // get tensors
  input_data = opTensors[0];
  scale = opTensors[1];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void BroadcastMulOpKernel::invoke() {
  int n = this->shape.at(0);
  int c = this->shape.at(1);
  int h = this->shape.at(2);
  int w = this->shape.at(3);
  for (int ni = 0; ni < n; ++ni) {
    for (int ci = 0; ci < c; ++ci) {
      for (int i = 0; i < h * w; ++i) {
        auto x = input_data->at(ni * c * h * w + ci * h * w + i);
        auto y = x * scale->at(ci);
        output_data->at(ni * c * h * w + ci * h * w + i) = y;
      }
    }
  }
  if (do_relu) {
    relu(output_data->data(), output_data->data(), output_data->size());
  }
  if (datatype == DataType::INT8) {
    for (size_t i = 0; i < output_data->size(); ++i) {
      output_data->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
          output_data->at(i), (uint32_t)rshift, (uint32_t)mutliplier, true);
    }
  } else if (datatype == DataType::BF16) {
    clean16bitmantissa(output_data->data(), output_data->data(),
                       output_data->size());
  }
}
void BroadcastMulOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " BroadcastMul op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};
std::vector<float> BroadcastMulOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}
void BroadcastMulOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir