#include "tpuc/Interpreter/cpu/shuffle_channel.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {

ShuffleChannelOpKernel::ShuffleChannelOpKernel(Operation &op,
                                               value_map_t &valueMapping) {
  auto shufflechannelOp = cast<tpu::ShuffleChannelOp>(op);
  assert(shufflechannelOp);
  llvm::outs() << " ShuffleChannelOp op: [" << shufflechannelOp.name() << "]\n";

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = shufflechannelOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  llvm::outs() << "    =>required memory size: [" << size << "]\n";
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  auto input_type =
      shufflechannelOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();

  this->name = shufflechannelOp.name().str();
  this->group = shufflechannelOp.group();

  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());

  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void ShuffleChannelOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " ShuffleChannelOp op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> ShuffleChannelOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
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

void ShuffleChannelOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir