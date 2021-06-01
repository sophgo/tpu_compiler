#ifndef INTERPRETER_CPU_NORMALIZE_H
#define INTERPRETER_CPU_NORMALIZE_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class NormalizeOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUNormlizeOp";

  NormalizeOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData scale_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
  SyncedDataShape scale_shape;
  // param
  bool across_spatial;
  bool channel_shared;
};
} // namespace mlir

#endif