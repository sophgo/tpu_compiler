#ifndef INTERPRETER_CPU_REFLECTION_PAD1D_H
#define INTERPRETER_CPU_REFLECTION_PAD1D_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {

class ReflectionPadOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUReflectionPadOp";

  ReflectionPadOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void dump() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  // param
  std::vector<int> pad;
};
} // namespace mlir
#endif
