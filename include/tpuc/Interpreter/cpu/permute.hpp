#ifndef INTERPRETER_CPU_PERMUTE_H
#define INTERPRETER_CPU_PERMUTE_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {
void permute(float *src, float *dst, const std::vector<int64_t> &input_shape,
             std::vector<unsigned int> &order);
class PermuteOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUPermuteOp";

  PermuteOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  // param
  std::vector<unsigned int> order;
};
} // namespace mlir
#endif