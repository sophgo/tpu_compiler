#ifndef INTERPRETER_CPU_BROADCASTMUL_H
#define INTERPRETER_CPU_BROADCASTMUL_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class BroadcastMulOpKernel : public CPUOpKernel<BroadcastMulOpKernel> {
public:
  static constexpr const char *OpName = "CPUBroadcastMulOp";

  BroadcastMulOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  SyncedData input_data;
  SyncedData scale;
  SyncedData output_data;
  SyncedDataShape input_shape;
  SyncedDataShape scale_shape;
  // param
  bool do_relu;
  float rshift;
  float mutliplier;
};
} // namespace mlir

#endif