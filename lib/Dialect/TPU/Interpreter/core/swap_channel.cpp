#include "tpuc/Interpreter/cpu/swap_channel.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {

SwapChannelOpKernel::SwapChannelOpKernel(Operation &op,
                                         value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto swapchannelOp = cast<tpu::SwapChannelOp>(op);
  auto input_type = swapchannelOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();
  arrayAttrToVector(swapchannelOp.channel_order().getValue(), order);
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
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

} // namespace mlir