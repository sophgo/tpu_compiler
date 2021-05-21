#include "tpuc/Interpreter/cpu/upsample.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {

UpsampleOpKernel::UpsampleOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto upsampleOp = cast<tpu::UpsampleOp>(op);
  auto input_type = upsampleOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();
  this->scale_h = upsampleOp.scale_h();
  this->scale_w = upsampleOp.scale_w();
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void UpsampleOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " UpsampleOp op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> UpsampleOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void UpsampleOpKernel::invoke() {
  int n = shape[0];
  int c = shape[1];
  int ih = input_shape[2];
  int iw = input_shape[3];
  int h = ih * scale_h;
  int w = iw * scale_w;
  for (int ni = 0; ni < n; ni++) {
    for (int ci = 0; ci < c; ci++) {
      for (int hi = 0; hi < h; hi++) {
        for (int wi = 0; wi < w; wi++) {
          int nwi = wi / scale_w;
          int nhi = hi / scale_h;
          int out_idx = (((ni * c + ci) * h) + hi) * w + wi;
          int in_idx =
              (((ni * c + ci) * (h / scale_h)) + nhi) * (w / scale_w) + nwi;
          output_data->at(out_idx) = input_data->at(in_idx);
        }
      }
    }
  }
}
void UpsampleOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir