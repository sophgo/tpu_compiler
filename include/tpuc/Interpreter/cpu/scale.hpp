#ifndef INTERPRETER_CPU_SCALE_H
#define INTERPRETER_CPU_SCALE_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class ScaleOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUScaleOp";

  ScaleOpKernel(Operation &op, value_map_t &valueMapping,
                weight_map_t &weightMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData scale;
  SyncedData bias;
  SyncedData output_data;
  SyncedDataShape input_shape;
  SyncedDataShape bias_shape;
  SyncedDataShape scale_shape;
  // param
};
} // namespace mlir

#endif