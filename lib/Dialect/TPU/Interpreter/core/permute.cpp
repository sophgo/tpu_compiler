#include "tpuc/Interpreter/cpu/permute.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {

void permute(float *src, float *dst, const std::vector<int64_t> &ishape,
             std::vector<int> &order) {
  int in = ishape[0];
  int ic = ishape[1];
  int ih = ishape[2];
  int iw = ishape[3];
  int size = std::accumulate(ishape.begin(), ishape.end(), 1,
                             std::multiplies<int64_t>());
  std::vector<float> tmp_data(size);
  std::memcpy(tmp_data.data(), src, size * sizeof(float));
  for (int n = 0; n < in; n++) {
    for (int c = 0; c < ic; c++) {
      for (int h = 0; h < ih; h++) {
        for (int w = 0; w < iw; w++) {
          int cur[4] = {n, c, h, w};
          int in_idx = w + h * iw + c * ih * iw + n * ic * ih * iw;
          int out_idx = cur[order[3]] + cur[order[2]] * ishape[order[3]] +
                        cur[order[1]] * ishape[order[3]] * ishape[order[2]] +
                        cur[order[0]] * ishape[order[3]] * ishape[order[2]] *
                            ishape[order[1]];
          dst[out_idx] = tmp_data[in_idx];
        }
      }
    }
  }
};

PermuteOpKernel::PermuteOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto permuteOp = cast<tpu::PermuteOp>(op);
  auto input_type = permuteOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();
  arrayAttrToVector(permuteOp.order(), this->order);
  parsePermuteParam(input_shape, order, shape_4, order_4);
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void PermuteOpKernel::invoke() {
  permute(input_data->data(), output_data->data(), shape_4, order_4);
}

} // namespace mlir