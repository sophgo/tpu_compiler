#ifndef INTERPRETER_CPU_QUADRATICSUM_H
#define INTERPRETER_CPU_QUADRATICSUM_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class QuadraticSumOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUQuadraticSumOp";

  QuadraticSumOpKernel(Operation &op, value_map_t &valueMapping,
                       weight_map_t &weightMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
};

} // namespace mlir

#endif
