#ifndef INTERPRETER_CPU_SCALE_LUT_H
#define INTERPRETER_CPU_SCALE_LUT_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class ScaleLutOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUScaleLutOp";

  ScaleLutOpKernel(Operation &op, value_map_t &valueMapping,
                   weight_map_t &weightMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData table;
  SyncedData output_data;
  SyncedDataShape input_shape;
  SyncedDataShape table_shape;
  // param
};
} // namespace mlir

#endif