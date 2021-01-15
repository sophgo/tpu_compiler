#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpukernel.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {
InputOpKernel::InputOpKernel(
    Operation &op, value_map_t &valueMapping,
    std::vector<std::pair<std::string, size_t>> &input_details) {
  auto inputOp = dyn_cast<tpu::InputOp>(op);
  llvm::outs() << " Input op: [" << inputOp.name() << "]\n";

  auto result = inputOp.getResult();
  auto type = result.getType().cast<TensorType>();
  int64_t size = getTensorSize(result);
  llvm::outs() << "    =>required memory size: [" << size << "]\n";
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  this->shape = type.getShape();

  this->name = inputOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  input_details.push_back(std::make_pair(name, size));

  set_datatype(getOpQuant(&op).str());
  this->data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void InputOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->data->capacity()) {
    llvm::errs() << " Input op: [" << this->name
                 << "] required memsize :" << this->data->capacity() << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->data->assign(data.begin(), data.end());
};
std::vector<float> InputOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->data->begin(), this->data->end());
  return ret;
}
void InputOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir