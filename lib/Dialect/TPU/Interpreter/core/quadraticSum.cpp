#include "tpuc/Interpreter/cpu/quadraticSum.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/conv.hpp"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/QuantizationArithmetic.h"

namespace mlir {

QuadraticSumOpKernel::QuadraticSumOpKernel(Operation &op,
                                           value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {

  this->input_shape = getTensorShape(op.getOperand(0));
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
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