#ifndef INTERPRETER_CPU_PRIORBOX_H
#define INTERPRETER_CPU_PRIORBOX_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class PriorBoxOpKernel : public CPUOpKernel<PriorBoxOpKernel> {
public:
  static constexpr const char *OpName = "CPUPriorBoxOp";

  PriorBoxOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData output_data;
  SyncedDataShape input_shape;
  SyncedDataShape input_image_shape;

  // param
  std::vector<float> min_size;
  std::vector<float> max_size;
  std::vector<float> aspect_ratios;
  std::vector<float> variance;

  bool clip;
  bool use_default_aspect_ratio;
  float offset;
  float step_w;
  float step_h;
  int num_priors;
  int img_width;
  int img_height;
};
} // namespace mlir

#endif
