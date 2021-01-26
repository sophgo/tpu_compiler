#ifndef INTERPRETER_CPU_REVERSE_H
#define INTERPRETER_CPU_REVERSE_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {
class ReverseOpKernel : public CPUOpKernel<ReverseOpKernel> {
public:
  static constexpr const char *OpName = "CPUReverseOp";

  ReverseOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  // param
  int axis;
};
} // namespace mlir
#endif