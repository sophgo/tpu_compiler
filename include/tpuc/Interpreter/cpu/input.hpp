#ifndef INTERPRETER_CPU_INPUT_H
#define INTERPRETER_CPU_INPUT_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {

class InputOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUInputOp";
  InputOpKernel(Operation &op, value_map_t &valueMapping,
                weight_map_t &weightMapping,
                std::vector<std::pair<std::string, size_t>> &input_details);
  void invoke() override{
      // input op no need to invoke, skip
  };
  void set_tensor(const std::vector<float> &data) override;

private:
  SyncedData data;
};

} // namespace mlir
#endif