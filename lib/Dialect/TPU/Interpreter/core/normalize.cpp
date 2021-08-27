#include "tpuc/Interpreter/cpu/normalize.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/MlirModuleInterpreter.h"

namespace mlir {
NormalizeOpKernel::NormalizeOpKernel(Operation &op, value_map_t &valueMapping,
                                     weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto normOp = cast<tpu::NormalizeOp>(op);
  this->across_spatial = normOp.across_spatial();
  this->channel_shared = normOp.channel_shared();
  // get tensors
  input_data = this->opdTensors[0];
  scale_data = this->opdTensors[1];
  output_data = this->resTensor;
}

void NormalizeOpKernel::invoke() {

  int n = this->shape.at(0);
  int c = this->shape.at(1);
  int h = this->shape.size() > 2 ? this->shape.at(2) : 1;
  int w = this->shape.size() > 3 ? this->shape.at(3) : 1;

  float eps = 1.0e-5;
  if (!across_spatial) { // only ssd case currently
    auto spatial_dim = h * w;
    auto norm_ = c * h * w;
    for (int ni = 0; ni < n; ni++)
      for (int i = 0; i < spatial_dim; i++) {
        auto value = 0;
        for (int ci = 0; ci < c; ci++) {
          value += pow(input_data->at(ni * norm_ + ci * spatial_dim + i), 2);
        }
        for (int ci = 0; ci < c; ci++) {
          output_data->at(ni * norm_ + ci * spatial_dim + i) =
              (input_data->at(ni * norm_ + ci * spatial_dim + i) /
               sqrt(value + eps)) *
              scale_data->at(ci);
        }
      }
  }
}

} // namespace mlir