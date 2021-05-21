#ifndef INTERPRETER_CPU_BATCHNORM_H
#define INTERPRETER_CPU_BATCHNORM_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class BatchNormOpKernel : public CPUOpKernel<BatchNormOpKernel> {
public:
  static constexpr const char *OpName = "CPUBatchNormOp";

  BatchNormOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData mean;
  SyncedData variance;
  SyncedData scale;
  SyncedData output_data;
  SyncedDataShape input_shape;
  SyncedDataShape output_shape;
  SyncedDataShape mean_shape;
  SyncedDataShape variance_shape;
  SyncedDataShape scale_shape;
  // param
  float variance_epsilon;
};
} // namespace mlir

#endif