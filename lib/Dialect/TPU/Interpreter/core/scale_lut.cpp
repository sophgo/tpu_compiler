#include "tpuc/Interpreter/cpu/scale_lut.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {
ScaleLutOpKernel::ScaleLutOpKernel(Operation &op, value_map_t &valueMapping) {
  auto castOp = cast<tpu::ScaleLutOp>(op);
  assert(castOp);
  LLVM_DEBUG(llvm::outs() << " ScaleLutOp: [" << castOp.name() << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = castOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  this->name = castOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());

  // get tensors
  input_data = opTensors[0];
  table = opTensors[1];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}

void ScaleLutOpKernel::invoke() {
  int n = this->shape.at(0);
  int c = this->shape.at(1);
  int h = this->shape.at(2);
  int w = this->shape.at(3);
  for (int ni = 0; ni < n; ++ni) {
    for (int ci = 0; ci < c; ++ci) {
      for (int i = 0; i < h * w; ++i) {
        int index = ni * c * h * w + ci * h * w + i;
        auto x = input_data->at(index);
        auto y = table->at((int)(ci * 256 + x));
        output_data->at(index) = y;
      }
    }
  }
}

void ScaleLutOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " Scale op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> ScaleLutOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}
void ScaleLutOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir