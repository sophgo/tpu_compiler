#include "tpuc/Interpreter/cpu/crop.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {

inline int crop_offset(const std::vector<int> &indices, long int *shape) {
  int offset = 0;
  for (int i = 0; i < 4; ++i) {
    offset *= shape[i];
    if ((int)indices.size() > i) {
      offset += indices[i];
    }
  }
  return offset;
}

void crop(float *input, float *output, long int *input_shape,
          long int *output_shape, int cur_dim, int *offsets, int *indices) {
  // for loop if dim is not last
  if (cur_dim + 1 < 4) {
    for (int i = 0; i < output_shape[cur_dim]; ++i) {
      indices[cur_dim] = i;
      crop(input, output, input_shape, output_shape, cur_dim + 1, offsets,
           indices);
    }
  } else {
    std::vector<int> ind_red(cur_dim, 0);
    std::vector<int> ind_off(cur_dim + 1, 0);

    for (int j = 0; j < cur_dim; ++j) {
      ind_red[j] = indices[j];

      ind_off[j] = indices[j] + offsets[j];
    }
    ind_off[cur_dim] = offsets[cur_dim];

    std::memcpy(output + crop_offset(ind_red, output_shape),
                input + crop_offset(ind_off, input_shape),
                sizeof(float) * output_shape[cur_dim]);
  }
};

CropOpKernel::CropOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto cropOp = cast<tpu::CropOp>(op);
  auto input_type = cropOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();
  arrayAttrToVector(cropOp.crop_offset().getValue(), this->crop_offset);
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void CropOpKernel::invoke() {
  std::vector<int> indices(input_shape.size(), 0);
  std::vector<int64_t> crop_shape(this->shape.begin(), this->shape.end());
  crop(input_data->data(), output_data->data(), input_shape.data(),
       crop_shape.data(), 0, crop_offset.data(), indices.data());
}

} // namespace mlir