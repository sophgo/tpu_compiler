#ifndef INTERPRETER_CPU_INTERPOLATION_H
#define INTERPRETER_CPU_INTERPOLATION_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {

class InterpolationOpKernel : public CPUOpKernel<InterpolationOpKernel> {
public:
  static constexpr const char *OpName = "CPUInterpolationOp";

  InterpolationOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  // param
  int shrink_factor = 0;
  int zoom_factor = 0;
  int height = 0;
  int width = 0;
  int pad_beg;
  int pad_end;
  std::string coordinate_transformation_mode;
};
} // namespace mlir
#endif