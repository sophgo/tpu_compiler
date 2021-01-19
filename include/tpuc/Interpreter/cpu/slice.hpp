#ifndef INTERPRETER_CPU_SLICE_H
#define INTERPRETER_CPU_SCALE_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class SliceOpKernel : public CPUOpKernel<SliceOpKernel> {
public:
  static constexpr const char *OpName = "CPUSliceOp";

  SliceOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

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