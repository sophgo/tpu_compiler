#ifndef INTERPRETER_CPU_ZEROMASK_H
#define INTERPRETER_CPU_ZEROMASK_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {
class ZeroMaskOpKernel : public CPUOpKernel<ZeroMaskOpKernel> {
public:
  static constexpr const char *OpName = "CPUZeroMaskOp";

  ZeroMaskOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
};
} // namespace mlir
#endif