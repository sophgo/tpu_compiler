#ifndef INTERPRETER_CPU_COPY_H
#define INTERPRETER_CPU_COPY_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {
class CopyOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUCopyOp";

  CopyOpKernel(Operation &op, value_map_t &valueMapping,
                  weight_map_t &weightMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;

  // param
  std::vector<int> input_stride;
  std::vector<int> output_stride;
  std::vector<int> copy_shape;
};
} // namespace mlir
#endif