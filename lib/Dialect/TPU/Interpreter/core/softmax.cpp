#include "tpuc/Interpreter/cpu/softmax.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"

namespace mlir {

SoftmaxOpKernel::SoftmaxOpKernel(Operation &op, value_map_t &valueMapping,
                                 bool cpu)
    : CPUOpKernel(op, valueMapping) {
  if (cpu) {
    auto castOp = cast<tpu::SoftmaxCpuOp>(op);
    this->axis = castOp.axis();
  } else {
    auto castOp = cast<tpu::SoftmaxOp>(op);
    this->axis = castOp.axis();
  }
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void SoftmaxOpKernel::set_tensor(const std::vector<float> &data) {
  this->input_data->assign(data.begin(), data.end());
};
std::vector<float> SoftmaxOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}
void SoftmaxOpKernel::invoke() {
  if (this->shape.size() == 2) {
    bool isBF16 = datatype == DataType::BF16;
    int ret = my_softmax2D(input_data->data(), output_data->data(),
                           this->shape.at(0), this->shape.at(1), isBF16);
    assert(ret == 0);
  } else if (this->shape.size() == 3) {
    bool isBF16 = datatype == DataType::BF16;
    int ret = my_softmax3D(input_data->data(), output_data->data(), this->axis,
                           this->shape, isBF16);
    assert(ret == 0);
  } else if (this->shape.size() == 4) {
    bool isBF16 = datatype == DataType::BF16;
    int ret = my_softmax4D(input_data->data(), output_data->data(), this->axis,
                           this->shape, isBF16);
    assert(ret == 0);
  } else {
    OpKernel::dump();
    llvm_unreachable("TODO");
  }
}

void SoftmaxOpKernel::dump() { OpKernel::dump(); };
} // namespace mlir