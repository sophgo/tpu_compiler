#ifndef INTERPRETER_CPU_FULLYCONNECTED_H
#define INTERPRETER_CPU_FULLYCONNECTED_H

#include "mkldnn.hpp"
#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {

class FullyConnectedOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUFullyConnectedOp";

  FullyConnectedOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData filter_data;
  SyncedData bias_data;
  SyncedData output_data;
  SyncedData rshift_data;
  SyncedData multiplier_data;
  SyncedDataShape input_shape;
  SyncedDataShape filter_shape;
  SyncedDataShape bias_shape;

  // param
  int batch;
  int m;
  int k;
  int n;
  bool do_relu = false;
};
} // namespace mlir

#endif