#include "tpuc/Interpreter/cpu/reverse.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/MlirModuleInterpreter.h"

namespace mlir {

ReverseOpKernel::ReverseOpKernel(Operation &op, value_map_t &valueMapping,
                                 weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto reverseOp = cast<tpu::ReverseOp>(op);
  auto input_type = reverseOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();
  this->axis = reverseOp.axis();
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void ReverseOpKernel::invoke() {
  int on = shape[0];
  int oc = shape[1];
  int oh = shape.size() > 2 ? shape[2] : 1;
  int ow = shape.size() > 3 ? shape[3] : 1;
  int dim[] = {on, oc, oh, ow};
  int stride[] = {oc * oh * ow, oh * ow, ow, 1};
  for (int in = 0; in < on; in++) {
    for (int ic = 0; ic < oc; ic++) {
      for (int ih = 0; ih < oh; ih++) {
        for (int iw = 0; iw < ow; iw++) {
          int src_index[] = {in, ic, ih, iw};
          int dst_index[] = {in, ic, ih, iw};
          dst_index[axis] = dim[axis] - src_index[axis] - 1;
          int src_offset = src_index[0] * stride[0] + src_index[1] * stride[1] +
                           src_index[2] * stride[2] + src_index[3] * stride[3];
          int dst_offset = dst_index[0] * stride[0] + dst_index[1] * stride[1] +
                           dst_index[2] * stride[2] + dst_index[3] * stride[3];
          output_data->at(dst_offset) = input_data->at(src_offset);
        }
      }
    }
  }
}

} // namespace mlir