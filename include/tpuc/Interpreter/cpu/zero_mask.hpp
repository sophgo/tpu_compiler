#ifndef INTERPRETER_CPU_ZEROMASK_H
#define INTERPRETER_CPU_ZEROMASK_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class ZeroMaskOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUZeroMaskOp";
  ZeroMaskOpKernel(Operation &op, value_map_t &valueMapping,
                    weight_map_t &weightMapping);
  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  bool positive;
};
} // namespace mlir

#endif
