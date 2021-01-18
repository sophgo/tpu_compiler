#include "tpuc/Interpreter/cpu/softmax.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"

namespace mlir {
SoftmaxOpKernel::SoftmaxOpKernel(Operation &op, value_map_t &valueMapping) {
  auto castOp = cast<tpu::SoftmaxOp>(op);
  assert(castOp);
  llvm::outs() << " Softmax op: [" << castOp.name() << "]\n";

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = castOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  llvm::outs() << "    =>required memory size: [" << size << "]\n";
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  this->name = castOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());

  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
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
  } else {
    llvm_unreachable("TODO");
  }
};
void SoftmaxOpKernel::dump() { OpKernel::dump(); };
} // namespace mlir