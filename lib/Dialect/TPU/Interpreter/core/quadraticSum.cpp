#include "tpuc/Interpreter/cpu/quadraticSum.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/conv.hpp"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/QuantizationArithmetic.h"

// Quantize an Activation tensor into INT8, given threshold

namespace mlir {
QuadraticSumOpKernel::QuadraticSumOpKernel(Operation &op,
                                           value_map_t &valueMapping) {
  auto quantOp = cast<tpu::QuadraticSumOp>(op);
  assert(quantOp);
  LLVM_DEBUG(llvm::outs() << " Quant op: [" << quantOp.name() << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = quantOp.getResult();
  auto size = getTensorSize(result);
  auto output_dataensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);

  this->shape = getTensorShape(result);
  this->input_shape = getTensorShape(op.getOperand(0));
  this->name = quantOp.name().str();
  this->op_type = op.getName().getStringRef().str();

  set_datatype(getOpQuant(&op).str());
  // get tensors
  input_data = opTensors[0];
  output_data = output_dataensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(output_dataensor);
}

void QuadraticSumOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " Quant op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> QuadraticSumOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void QuadraticSumOpKernel::invoke() {
  int n = input_shape[0];
  int c = input_shape[1];
  int h = input_shape[2];
  int w = input_shape[3];
  once_mkldnn_conv(input_data->data(), input_data->data(), nullptr,
                   output_data->data(), n, c, h, w, c, 1, 1, h, w, h, w, 1, 1,
                   0, 0, 0, 0, c, 0);
  if (datatype == DataType::BF16) {
    clean16bitmantissa(output_data->data(), output_data->data(),
                       output_data->size());
  }
}

void QuadraticSumOpKernel::dump() { OpKernel::dump(); }

} // namespace mlir