#ifndef INTERPRETER_CPU_POOLMASK_H
#define INTERPRETER_CPU_POOLMASK_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {
class PoolMaskOpKernel : public CPUOpKernel<PoolMaskOpKernel> {
public:
  static constexpr const char *OpName = "CPUPoolMaskOp";

  PoolMaskOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
  int scale;
};
} // namespace mlir
#endif