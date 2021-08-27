#ifndef INTERPRETER_CPU_CLIP_H
#define INTERPRETER_CPU_CLIP_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {
void crop(float *input, float *output, long int *input_shape,
          long int *output_shape, int cur_dim, int *offsets, int *indices);

class ClipOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUClipOp";

  ClipOpKernel(Operation &op, value_map_t &valueMapping,
               weight_map_t &weightMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  // int8
  float rshift;
  float multiplier;
  // param
  float min;
  float max;
};
} // namespace mlir
#endif