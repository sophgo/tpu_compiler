#ifndef INTERPRETER_CPU_SWAP_CHANNEL_H
#define INTERPRETER_CPU_SWAP_CHANNEL_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {
class SwapChannelOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUSwapChannelOp";

  SwapChannelOpKernel(Operation &op, value_map_t &valueMapping,
                      weight_map_t &weightMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
  std::vector<int> order;
};
} // namespace mlir
#endif