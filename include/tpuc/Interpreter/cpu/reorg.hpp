#ifndef INTERPRETER_CPU_REORG_H
#define INTERPRETER_CPU_REORG_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {
class ReorgOpKernel : public CPUOpKernel<ReorgOpKernel> {
public:
  static constexpr const char *OpName = "CPUReorgOp";

  ReorgOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  // param
  int stride;
};
} // namespace mlir
#endif