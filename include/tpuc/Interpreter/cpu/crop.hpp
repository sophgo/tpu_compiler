#ifndef INTERPRETER_CPU_CROP_H
#define INTERPRETER_CPU_CROP_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {
void crop(float *input, float *output, long int *input_shape,
          long int *output_shape, int cur_dim, int *offsets, int *indices);

class CropOpKernel : public CPUOpKernel<CropOpKernel> {
public:
  static constexpr const char *OpName = "CPUCropOp";

  CropOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  // param
  std::vector<int> crop_offset;
};
} // namespace mlir
#endif