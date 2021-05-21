#ifndef INTERPRETER_CPU_SHUFFLE_CHANNEL_H
#define INTERPRETER_CPU_SHUFFLE_CHANNEL_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {
class ShuffleChannelOpKernel : public CPUOpKernel<ShuffleChannelOpKernel> {
public:
  static constexpr const char *OpName = "CPUShuffleChannelOp";

  ShuffleChannelOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
  int group;
};
} // namespace mlir
#endif