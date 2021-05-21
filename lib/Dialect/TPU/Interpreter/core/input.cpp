#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpukernel.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {

InputOpKernel::InputOpKernel(
    Operation &op, value_map_t &valueMapping,
    std::vector<std::pair<std::string, size_t>> &input_details)
    : CPUOpKernel(op, valueMapping, false) {

  input_details.push_back(std::make_pair(this->name, this->resTensor->size()));
  this->data = this->resTensor;
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