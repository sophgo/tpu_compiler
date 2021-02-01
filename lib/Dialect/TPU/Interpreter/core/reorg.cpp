#include "tpuc/Interpreter/cpu/reorg.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {

ReorgOpKernel::ReorgOpKernel(Operation &op, value_map_t &valueMapping) {
  auto reorgOp = cast<tpu::ReorgOp>(op);
  assert(reorgOp);
  LLVM_DEBUG(llvm::outs() << " ReorgOp op: [" << reorgOp.name() << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = reorgOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  auto input_type = reorgOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();

  this->name = reorgOp.name().str();
  this->stride = reorgOp.stride();
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());
  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void ReorgOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " ReorgOp op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> ReorgOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void ReorgOpKernel::invoke() {
  int n = input_shape[0];
  int c = input_shape[1];
  int h = input_shape[2];
  int w = input_shape[3];
  int out_c = c / (stride * stride);
  int out_w = w * stride;
  int out_h = h * stride;
  for (int b = 0; b < n; b++) {
    for (int k = 0; k < c; k++) {
      for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
          int in_index = i + w * (j + h * (k + c * b));
          int c2 = k % out_c;
          int offset = k / out_c;
          int w2 = i * stride + offset % stride;
          int h2 = j * stride + offset / stride;
          int out_index = w2 + out_w * (h2 + out_h * (c2 + out_c * b));
          output_data->at(in_index) = input_data->at(out_index);
        }
      }
    }
  }
}
void ReorgOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir