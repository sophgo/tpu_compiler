#include "tpuc/Interpreter/cpu/crop.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/MlirModuleInterpreter.h"

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

CropOpKernel::CropOpKernel(Operation &op, value_map_t &valueMapping,
                           weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto cropOp = cast<tpu::CropOp>(op);
  this->input_shape = getTensorShape(cropOp.input());
  this->output_shape = getTensorShape(cropOp.output());
  arrayAttrToVector(cropOp.crop_offset(), this->crop_offset);
  assert(crop_offset.size() == shape.size());
  if (cropOp.steps().hasValue()) {
    arrayAttrToVector(cropOp.steps().getValue(), this->steps);
  }
  for (uint32_t i = steps.size(); i < shape.size(); i++) {
    steps.push_back(1);
  }

  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void CropOpKernel::invoke() {
  int32_t total_steps = std::accumulate(steps.begin(), steps.end(), 1,
                                        std::multiplies<int32_t>());
  if (total_steps == 1) {
    std::vector<int> indices(input_shape.size(), 0);
    std::vector<int64_t> crop_shape(this->shape.begin(), this->shape.end());
    crop(0, crop_offset.data(), indices.data());
  } else {
    for (uint32_t i = steps.size(); i < 4; i++) {
      steps.push_back(1);
      input_shape.push_back(1);
      output_shape.push_back(1);
      crop_offset.push_back(1);
    }
    int n = output_shape[0];
    int c = output_shape[1];
    int h = output_shape[2];
    int w = output_shape[3];
    int in, ic, ih, iw;
    for (int on = 0; on < n; on++) {
      for (int oc = 0; oc < c; oc++) {
        for (int oh = 0; oh < h; oh++) {
          for (int ow = 0; ow < w; ow++) {
            int o_offset = on * c * h * w + oc * h * w + oh * w + ow;
            in = on * steps[0] + crop_offset[0];
            ic = oc * steps[1] + crop_offset[1];
            ih = oh * steps[2] + crop_offset[2];
            iw = ow * steps[3] + crop_offset[3];
            int i_offset =
                in * input_shape[1] * input_shape[2] * input_shape[3] +
                ic * input_shape[2] * input_shape[3] + ih * input_shape[3] + iw;
            output_data->at(o_offset) = input_data->at(i_offset);
          }
        }
      }
    }
  }
}

} // namespace mlir