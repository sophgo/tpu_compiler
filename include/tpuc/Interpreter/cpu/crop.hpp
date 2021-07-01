#ifndef INTERPRETER_CPU_CROP_H
#define INTERPRETER_CPU_CROP_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {

class CropOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUCropOp";

  CropOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  void crop(int cur_dim, int *offsets, int *indices);
  int offset(const std::vector<int> &indices, const SyncedDataShape &shape);
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  // param
  std::vector<int> crop_offset;
};
} // namespace mlir
#endif