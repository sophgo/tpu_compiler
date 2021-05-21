#include "tpuc/Interpreter/cpu/dilate.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"

namespace mlir {
int calc_dilute_hw(int h, int ins_h, int ins_h_l, int pad_h_b, int pad_h_t) {
  return (h - 1) * (ins_h + 1) + ins_h_l + 1 + pad_h_t + pad_h_b;
}

void dilateActivation(float *input, float *output, int pad_h_t, int pad_h_b,
                      int ins_h, int ins_h_l, int pad_w_l, int pad_w_r,
                      int ins_w, int ins_w_l, int n, int c, int h, int w,
                      int fill_constant) {
  int oh = calc_dilute_hw(h, ins_h, ins_h_l, pad_h_t, pad_h_b);
  int ow = calc_dilute_hw(w, ins_w, ins_w_l, pad_w_l, pad_w_r);
  assert(!ins_h_l && !ins_w_l);
  for (int in = 0; in < n; in++) {
    for (int ic = 0; ic < c; ic++) {
      for (int _oh = 0; _oh < oh; _oh++) {
        for (int _ow = 0; _ow < ow; _ow++) {
          int out_idx = (((in * c + ic) * oh) + _oh) * ow + _ow;
          int in_nc = (in * c + ic) * h * w;
          output[out_idx] = fill_constant; // dilate
          if (_ow % (ins_w + 1) == 0 && _oh % (ins_h + 1) == 0) {
            output[out_idx] =
                input[in_nc + (_oh / (ins_h + 1)) * w + _ow / (ins_w + 1)];
          }
        }
      }
    }
  }
}
DilateOpKernel::DilateOpKernel(Operation &op, value_map_t &valueMapping)
  : CPUOpKernel(op, valueMapping) {
  auto dilateOp = cast<tpu::DilateOp>(op);
  auto input_type = dilateOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();

  std::vector<int32_t> ins;
  arrayAttrToVector(dilateOp.ins().getValue(), ins);

  this->ins_w = ins.size() > 0 ? ins[0] : 0;
  this->ins_h = ins.size() > 1 ? ins[1] : 0;
  this->fill_constant = dilateOp.fill_constant();
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void DilateOpKernel::invoke() {
  int n = this->input_shape.at(0);
  int ic = this->input_shape.at(1);
  int ih = this->input_shape.at(2);
  int iw = this->input_shape.at(3);

  dilateActivation(input_data->data(), output_data->data(), 0, 0, ins_h, 0, 0,
                   0, ins_w, 0, n, ic, ih, iw, fill_constant);
}

} // namespace mlir