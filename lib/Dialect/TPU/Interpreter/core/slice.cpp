#include "tpuc/Interpreter/cpu/slice.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {
void slice(float *input, float *output, int axis, int offset,
           std::vector<int64_t> input_shape,
           std::vector<int64_t> output_shape) {
  int osz = 1;
  for (int i = 0; i < axis; i++) {
    osz *= input_shape[i];
  }
  int isz = 1;
  for (unsigned i = axis + 1; i < input_shape.size(); i++) {
    isz *= input_shape[i];
  }
  int axis_total_size = input_shape[axis];
  int axis_slice_size = output_shape[axis];

  for (int n = 0; n < osz; ++n) {
    int output_offset = n * axis_slice_size * isz;
    int input_offset = n * axis_total_size * isz + offset * isz;
    std::memcpy(output + output_offset, input + input_offset,
                sizeof(float) * axis_slice_size * isz);
  }
}

SliceOpKernel::SliceOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto sliceOp = cast<tpu::SliceOp>(op);
  auto input_type = sliceOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();
  this->axis = sliceOp.axis();
  this->offset = sliceOp.offset();
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void SliceOpKernel::invoke() {
  slice(input_data->data(), output_data->data(), axis, offset, input_shape,
        shape);
}

} // namespace mlir