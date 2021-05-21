#ifndef INTERPRETER_CPU_DEPTHTOSPACE_H
#define INTERPRETER_CPU_DEPTHTOSPACE_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {

class DepthToSpaceOpKernel : public CPUOpKernel<DepthToSpaceOpKernel> {
public:
  static constexpr const char *OpName = "CPUDepthToSpaceOp";

  DepthToSpaceOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  // param
  bool dcr_mode;
  int upscale_factor;
};
} // namespace mlir
#endif