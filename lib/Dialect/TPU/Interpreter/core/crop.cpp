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

CropOpKernel::CropOpKernel(Operation &op, value_map_t &valueMapping) {
  auto cropOp = cast<tpu::CropOp>(op);
  assert(cropOp);
  llvm::outs() << " CropOp op: [" << cropOp.name() << "]\n";

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = cropOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  llvm::outs() << "    =>required memory size: [" << size << "]\n";
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  auto input_type = cropOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();

  this->name = cropOp.name().str();
  arrayAttrToVector(cropOp.crop_offset().getValue(), this->crop_offset);
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());
  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void CropOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " CropOp op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> CropOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void CropOpKernel::invoke() {
  std::vector<int> indices(input_shape.size(), 0);
  std::vector<int64_t> crop_shape(this->shape.begin(), this->shape.end());
  crop(input_data->data(), output_data->data(), input_shape.data(),
       crop_shape.data(), 0, crop_offset.data(), indices.data());
}

void CropOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir