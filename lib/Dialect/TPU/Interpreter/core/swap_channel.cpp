#include "tpuc/Interpreter/cpu/swap_channel.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {

SwapChannelOpKernel::SwapChannelOpKernel(Operation &op,
                                         value_map_t &valueMapping) {
  auto swapchannelOp = cast<tpu::SwapChannelOp>(op);
  assert(swapchannelOp);
  LLVM_DEBUG(llvm::outs() << " SwapChannelOp op: [" << swapchannelOp.name()
                          << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = swapchannelOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  auto input_type = swapchannelOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();

  this->name = swapchannelOp.name().str();

  arrayAttrToVector(swapchannelOp.channel_order().getValue(), order);

  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());

  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void SwapChannelOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " SwapChannelOp op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> SwapChannelOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void SwapChannelOpKernel::invoke() {
  int64_t n = input_shape[0];
  int64_t c = input_shape[1];
  int64_t frame_size = input_shape[2] * input_shape[3];
  int batch_length = c * frame_size;

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; j++) {
      float *p_in =
          input_data->data() + i * batch_length + frame_size * order[j];
      float *p_out = output_data->data() + i * batch_length + frame_size * j;
      memcpy((void *)p_out, (void *)p_in, frame_size * sizeof(float));
    }
  }
}

void SwapChannelOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir