#ifndef INTERPRETER_CPU_SOFTMAX_H
#define INTERPRETER_CPU_SOFTMAX_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class SoftmaxOpKernel : public CPUOpKernel<SoftmaxOpKernel> {
public:
  static constexpr const char *OpName = "CPUSoftmaxOp";

  SoftmaxOpKernel(Operation &op, value_map_t &valueMapping, bool cpu=false);

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
