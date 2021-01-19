#include "tpuc/Interpreter/cpu/slice.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {

SliceOpKernel::SliceOpKernel(Operation &op, value_map_t &valueMapping) {
  auto sliceOp = cast<tpu::SliceOp>(op);
  assert(sliceOp);
  llvm::outs() << " SliceOp op: [" << sliceOp.name() << "]\n";

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = sliceOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  llvm::outs() << "    =>required memory size: [" << size << "]\n";
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  auto input_type = sliceOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();

  this->name = sliceOp.name().str();
  this->axis = sliceOp.axis();
  this->offset = sliceOp.offset();

  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());

  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void SliceOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " SliceOp op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> SliceOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void SliceOpKernel::invoke() {
  int osz = 1;
  for (int i = 0; i < axis; i++) {
    osz *= input_shape[i];
  }
  int isz = 1;
  for (unsigned i = axis + 1; i < input_shape.size(); i++) {
    isz *= input_shape[i];
  }
  int axis_total_size = input_shape[axis];
  int axis_slice_size = shape[axis];

  for (int n = 0; n < osz; ++n) {
    int output_offset = n * axis_slice_size * isz;
    int input_offset = n * axis_total_size * isz + offset * isz;
    std::memcpy(output_data->data() + output_offset,
                input_data->data() + input_offset,
                sizeof(float) * axis_slice_size * isz);
  }
}

void SliceOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir