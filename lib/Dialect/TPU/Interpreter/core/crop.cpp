#include "tpuc/Interpreter/cpu/crop.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {

int CropOpKernel::offset(const std::vector<int> &indices,
                         const SyncedDataShape &shape) {
  int offset = 0;
  for (uint32_t i = 0; i < shape.size(); ++i) {
    offset *= shape[i];
    if (indices.size() > i) {
      offset += indices[i];
    }
  }
  return offset;
}

void CropOpKernel::crop(int cur_dim, int *offsets, int *indices) {
  // for loop if dim is not last
  int num_dims = shape.size();
  if (cur_dim + 1 < num_dims) {
    for (uint32_t i = 0; i < shape[cur_dim]; ++i) {
      indices[cur_dim] = i;
      crop(cur_dim + 1, offsets, indices);
    }
  } else {
    std::vector<int> ind_red(cur_dim, 0);
    std::vector<int> ind_off(cur_dim + 1, 0);

    for (int j = 0; j < cur_dim; ++j) {
      ind_red[j] = indices[j];

      ind_off[j] = indices[j] + offsets[j];
    }
    ind_off[cur_dim] = offsets[cur_dim];

    std::memcpy(output_data->data() + offset(ind_red, shape),
                input_data->data() + offset(ind_off, input_shape),
                sizeof(float) * shape[cur_dim]);
  }
};

CropOpKernel::CropOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto cropOp = cast<tpu::CropOp>(op);
  auto input_type = cropOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();
  arrayAttrToVector(cropOp.crop_offset().getValue(), this->crop_offset);
  assert(crop_offset.size() == shape.size());
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void CropOpKernel::invoke() {
  std::vector<int> indices(input_shape.size(), 0);
  std::vector<int64_t> crop_shape(this->shape.begin(), this->shape.end());
  crop(0, crop_offset.data(), indices.data());
}

} // namespace mlir