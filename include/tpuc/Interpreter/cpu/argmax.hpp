#ifndef INTERPRETER_CPU_ARGMAX_H
#define INTERPRETER_CPU_ARGMAX_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {

class ArgMaxOpKernel : public CPUOpKernel<ArgMaxOpKernel> {
public:
  static constexpr const char *OpName = "CPUArgMaxOp";

  ArgMaxOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  // param
  int axis;
};
} // namespace mlir
#endif