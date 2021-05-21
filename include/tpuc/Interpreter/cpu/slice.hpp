#ifndef INTERPRETER_CPU_SLICE_H
#define INTERPRETER_CPU_SCALE_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {
void slice(float *input, float *output, int axis, int offset,
           std::vector<int64_t> input_shape, std::vector<int64_t> output_shape);
class SliceOpKernel : public CPUOpKernel<SliceOpKernel> {
public:
  static constexpr const char *OpName = "CPUSliceOp";

  SliceOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
  // param
  int axis;
  int offset;
};
} // namespace mlir

#endif