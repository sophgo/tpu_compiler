#include "tpuc/Interpreter/cpu/embedding.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {
EmbeddingOpKernel::EmbeddingOpKernel(Operation &op, value_map_t &valueMapping)
  : CPUOpKernel(op, valueMapping) {
  auto embeddingOp = cast<tpu::EmbeddingOp>(op);
  auto input_type = embeddingOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();

  auto table_type = embeddingOp.table().getType().template cast<TensorType>();
  this->table_shape = table_type.getShape();

  auto output_type = embeddingOp.output().getType().template cast<TensorType>();
  this->output_shape = output_type.getShape();
  // get tensors
  input_data = this->opdTensors[0];
  table_data = this->opdTensors[1];
  output_data = this->resTensor;
}

void EmbeddingOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " embeddingOp op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> EmbeddingOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void EmbeddingOpKernel::invoke() {
  auto input = input_data->data();
  auto table = table_data->data();
  auto output = output_data->data();

  auto feature_dim = table_shape.back();
  assert(output_shape.back() == feature_dim && "must be the same feature dim");
  size_t count = 1;
  for (auto s : input_shape) {
    count *= s;
  }
  for (size_t i = 0; i < count; i++) {
    auto index = (size_t)input[i];
    size_t table_offset = (size_t)index * feature_dim;
    auto out_offset = i * feature_dim;
    memcpy(output + out_offset, table + table_offset, feature_dim * sizeof(float));
  }
}
void EmbeddingOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir
