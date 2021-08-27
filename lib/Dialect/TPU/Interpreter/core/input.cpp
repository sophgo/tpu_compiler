#include "tpuc/Interpreter/cpu/input.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/MlirModuleInterpreter.h"

namespace mlir {

InputOpKernel::InputOpKernel(
    Operation &op, value_map_t &valueMapping, weight_map_t &weightMapping,
    std::vector<std::pair<std::string, size_t>> &input_details)
    : CPUOpKernel(op, valueMapping, weightMapping, false) {

  input_details.push_back(std::make_pair(this->name, this->resTensor->size()));
  this->data = this->resTensor;
}

void InputOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->data->size()) {
    llvm::errs() << " Input op: [" << this->name
                 << "] required memsize :" << this->data->size() << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->data->assign(data.begin(), data.end());
}

} // namespace mlir