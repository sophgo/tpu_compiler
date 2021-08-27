#include "tpuc/Interpreter/cpu/shuffle_channel.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/MlirModuleInterpreter.h"

namespace mlir {

ShuffleChannelOpKernel::ShuffleChannelOpKernel(Operation &op,
                                               value_map_t &valueMapping,
                                               weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto shufflechannelOp = cast<tpu::ShuffleChannelOp>(op);
  auto input_type =
      shufflechannelOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();
  this->group = shufflechannelOp.group();

  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void ShuffleChannelOpKernel::invoke() {
  int64_t n = input_shape[0];
  int64_t c = input_shape[1];
  int64_t frame_size = input_shape[2] * input_shape[3];

  int batch_length = frame_size * c;
  int group_column = int(c / group);
  if (c % group != 0) {
    llvm::errs() << "Error: Wrong group size, c=" << c << ", group =" << group;
    llvm_unreachable("wrong group");
  }

  for (int i = 0; i < n; ++i) {
    float *p_in = input_data->data() + i * batch_length;
    float *p_out = output_data->data() + i * batch_length;
    for (int j = 0; j < group; ++j) // 2
    {
      for (int k = 0; k < group_column; ++k) // 3
      {
        float *p_i = p_in + (j * group_column + k) * frame_size;
        float *p_o = p_out + (k * group + j) * frame_size;

        std::memcpy(p_o, p_i, frame_size * sizeof(float));
      }
    }
  }
}

} // namespace mlir