#include "tpuc/Interpreter/cpu/scale_lut.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/MlirModuleInterpreter.h"

namespace mlir {

ScaleLutOpKernel::ScaleLutOpKernel(Operation &op, value_map_t &valueMapping,
                                   weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  input_data = this->opdTensors[0];
  table = this->opdTensors[1];
  output_data = this->resTensor;
}

void ScaleLutOpKernel::invoke() {
  int n = this->shape.at(0);
  int c = this->shape.at(1);
  int h = this->shape.at(2);
  int w = this->shape.at(3);
  for (int ni = 0; ni < n; ++ni) {
    for (int ci = 0; ci < c; ++ci) {
      for (int i = 0; i < h * w; ++i) {
        int index = ni * c * h * w + ci * h * w + i;
        auto x = input_data->at(index);
        auto y = table->at((int)(ci * 256 + x));
        output_data->at(index) = y;
      }
    }
  }
}

} // namespace mlir