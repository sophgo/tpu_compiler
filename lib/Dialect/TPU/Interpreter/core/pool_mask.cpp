#include "tpuc/Interpreter/cpu/pool_mask.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {

PoolMaskOpKernel::PoolMaskOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto pool_maskOp = cast<tpu::PoolMaskOp>(op);
  auto input_type = pool_maskOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();
  this->scale = pool_maskOp.scale();
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void PoolMaskOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " PoolMaskOp op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> PoolMaskOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void PoolMaskOpKernel::invoke() {
  int n = input_shape[0];
  int c = input_shape[1];
  int h = input_shape[2];
  int w = input_shape[3];
  int h_ex = shape[2];
  int w_ex = shape[3];
  for (int n_idx = 0; n_idx < n * c; n_idx++) {
    for (int h_idx = 0; h_idx < h_ex; h_idx += scale) {
      for (int w_idx = 0; w_idx < w_ex; w_idx += scale) {
        int index = n_idx * h * w + h_idx * w + w_idx;
        float max = input_data->at(index);
        int out_index = n_idx * h_ex * w_ex + h_idx * w_ex + w_idx;
        int max_index = out_index;
        for (int pool_h = 0; pool_h < scale && (pool_h + h_idx < h); pool_h++) {
          for (int pool_w = 0; pool_w < scale && (pool_w + w_idx < w);
               pool_w++) {
            int pool_index = index + pool_h * w + pool_w;
            if (input_data->at(pool_index) > max) {
              max = input_data->at(pool_index);
              max_index = out_index + pool_h * w_ex + pool_w;
            }
          }
        }
        output_data->at(max_index) = 1.0f;
      }
    }
  }
}

void PoolMaskOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir