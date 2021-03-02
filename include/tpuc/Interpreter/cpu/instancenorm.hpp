#ifndef INTERPRETER_CPU_INSTANCENORM_H
#define INTERPRETER_CPU_INSTANCENORM_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class InstanceNormOpKernel : public CPUOpKernel<InstanceNormOpKernel> {
public:
  static constexpr const char *OpName = "CPUInstanceNormOp";

  InstanceNormOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  SyncedData input_data;
  SyncedData bias;
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