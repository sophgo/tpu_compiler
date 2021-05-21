#include "tpuc/Interpreter/cpu/permute.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {

void permute(float *src, float *dst, const std::vector<int64_t> &input_shape,
             std::vector<unsigned int> &order) {
  if (order.size() != 4) {
    llvm_unreachable("permute order number must be 4");
  }
  int in = input_shape[0];
  int ic = input_shape[1];
  int ih = input_shape[2];
  int iw = input_shape[3];
  std::vector<int> shape(input_shape.begin(), input_shape.end());
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  std::vector<float> tmp_data(size);
  std::memcpy(tmp_data.data(), src, size * sizeof(float));
  for (int n = 0; n < in; n++) {
    for (int c = 0; c < ic; c++) {
      for (int h = 0; h < ih; h++) {
        for (int w = 0; w < iw; w++) {
          int cur[4] = {n, c, h, w};
          int in_idx = w + h * iw + c * ih * iw + n * ic * ih * iw;
          int out_idx = cur[order[3]] + cur[order[2]] * shape[order[3]] +
                        cur[order[1]] * shape[order[3]] * shape[order[2]] +
                        cur[order[0]] * shape[order[3]] * shape[order[2]] *
                            shape[order[1]];
          dst[out_idx] = tmp_data[in_idx];
        }
      }
    }
  }
};

PermuteOpKernel::PermuteOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto permuteOp = cast<tpu::PermuteOp>(op);
  auto input_type = permuteOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();
  this->order = {
      permuteOp.order0(),
      permuteOp.order1(),
      permuteOp.order2(),
      permuteOp.order3(),
  };
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void PermuteOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " PermuteOp op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> PermuteOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void PermuteOpKernel::invoke() {
  permute(input_data->data(), output_data->data(), input_shape, order);
}

void PermuteOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir