#ifndef INTERPRETER_CPU_SWAP_CHANNEL_H
#define INTERPRETER_CPU_SWAP_CHANNEL_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {
class SwapChannelOpKernel : public CPUOpKernel<SwapChannelOpKernel> {
public:
  static constexpr const char *OpName = "CPUSwapChannelOp";

  SwapChannelOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
  std::vector<int> order;
};
} // namespace mlir
#endif