#include "tpuc/Interpreter/cpu/reverse.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {

ReverseOpKernel::ReverseOpKernel(Operation &op, value_map_t &valueMapping) {
  auto reverseOp = cast<tpu::ReverseOp>(op);
  assert(reverseOp);
  llvm::outs() << " ReverseOp op: [" << reverseOp.name() << "]\n";

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = reverseOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  llvm::outs() << "    =>required memory size: [" << size << "]\n";

  this->shape = getTensorShape(result);

  auto input_type = reverseOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();

  this->name = reverseOp.name().str();
  this->axis = reverseOp.axis();

  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());
  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void ReverseOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " ReverseOp op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> ReverseOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
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
void ReverseOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir