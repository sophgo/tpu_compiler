#include "tpuc/Interpreter/cpu/normalize.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {
NormalizeOpKernel::NormalizeOpKernel(Operation &op, value_map_t &valueMapping) {
  auto normOp = cast<tpu::NormalizeOp>(op);
  assert(normOp);
  LLVM_DEBUG(llvm::outs() << " Normalize op: [" << normOp.name() << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = normOp.getResult();
  auto size = getTensorSize(result);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto resultTensor = std::make_shared<std::vector<float>>(size);

  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  this->across_spatial = normOp.across_spatial();
  this->channel_shared = normOp.channel_shared();

  this->name = normOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  set_datatype("NONE");
  // get tensors
  input_data = opTensors[0];
  scale_data = opTensors[1];

  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void NormalizeOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " Normalize op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};
std::vector<float> NormalizeOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void NormalizeOpKernel::invoke() {

  int n = this->shape.at(0);
  int c = this->shape.at(1);
  int h = this->shape.size() > 2 ? this->shape.at(2) : 1;
  int w = this->shape.size() > 3 ? this->shape.at(3) : 1;

  float eps = 1.0e-5;
  if (!across_spatial) { // only ssd case currently
    auto spatial_dim = h * w;
    auto norm_ = c * h * w;
    for (int ni = 0; ni < n; ni++)
      for (int i = 0; i < spatial_dim; i++) {
        auto value = 0;
        for (int ci = 0; ci < c; ci++) {
          value += pow(input_data->at(ni * norm_ + ci * spatial_dim + i), 2);
        }
        for (int ci = 0; ci < c; ci++) {
          output_data->at(ni * norm_ + ci * spatial_dim + i) =
              (input_data->at(ni * norm_ + ci * spatial_dim + i) /
               sqrt(value + eps)) *
              scale_data->at(ci);
        }
      }
  }
}
void NormalizeOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir