#ifndef INTERPRETER_CPU_POOLMASK_H
#define INTERPRETER_CPU_POOLMASK_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {
class PoolMaskOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUPoolMaskOp";

  PoolMaskOpKernel(Operation &op, value_map_t &valueMapping,
                   weight_map_t &weightMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
  int scale;
};
} // namespace mlir
#endif