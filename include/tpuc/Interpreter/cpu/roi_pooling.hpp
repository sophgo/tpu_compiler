#ifndef INTERPRETER_CPU_ROIPOOLING_H
#define INTERPRETER_CPU_ROIPOOLING_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class ROIPoolingOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUROIPoolingOp";

  ROIPoolingOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData output_data;
  SyncedData input_data;
  SyncedData rois;
  SyncedDataShape input_shape;
  SyncedDataShape roi_shape;

  // param
  int pooled_h;
  int pooled_w;
  float spatial_scale;
};
} // namespace mlir

#endif
