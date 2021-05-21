#include "tpuc/Interpreter/cpu/pad.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {

void pad_constant(float *input, float *output,
                  std::vector<int64_t> &input_shape, std::vector<int> &pads,
                  float const_val) {
  int in = input_shape[0];
  int ic = input_shape[1];
  int ih = input_shape[2];
  int iw = input_shape[3];
  // int on = pads[0] + pads[4] + in;
  int oc = pads[1] + pads[5] + ic;
  int oh = pads[2] + pads[6] + ih;
  int ow = pads[3] + pads[7] + iw;

  int in_offset = 0;
  int out_offset = 0;
  auto pad_n_begin_size = pads[0] * oc * oh * ow;
  for (int in_idx = 0; in_idx < in; in_idx++) {
    in_offset = in_idx * ic * ih * iw;
    auto pad_c_begin_size = pads[1] * oh * ow;
    out_offset = pad_n_begin_size + pad_c_begin_size + in_idx * oc * oh * ow;
    for (int ic_idx = 0; ic_idx < ic; ic_idx++) {
      auto in_ic_offset = in_offset + ic_idx * ih * iw;
      auto out_oc_offset = out_offset + ic_idx * oh * ow;

      // padding h_top and h_bottom;
      int pad_h_size = oh * iw;
      std::vector<float> out_pad_h(pad_h_size, const_val);

      int pad_top_offset = pads[2] * iw;
      memcpy(out_pad_h.data() + pad_top_offset, input + in_ic_offset,
             ih * iw * sizeof(int));

      if ((pads[3] != 0) || (pads[7] != 0)) {
        int pad_hw_size = oh * ow;
        std::vector<float> out_pad_hw(pad_hw_size, const_val);

        for (int i = 0; i < oh; i++) {
          int offset = i * ow + pads[3];
          memcpy(out_pad_hw.data() + offset, out_pad_h.data() + i * iw,
                 iw * sizeof(int));
        }
        memcpy(output + out_oc_offset, out_pad_hw.data(),
               pad_hw_size * sizeof(int));
      } else {
        memcpy(output + out_oc_offset, out_pad_h.data(),
               pad_h_size * sizeof(int));
      }
    }
  }
}

PadOpKernel::PadOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto padOp = cast<tpu::PadOp>(op);
  auto input_type = padOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();
  this->const_val = padOp.const_val().convertToFloat();
  arrayAttrToVector(padOp.pads().getValue(), this->pads);
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void PadOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " PadOp op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> PadOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void PadOpKernel::invoke() {
  int on = pads[0] + pads[4] + input_shape[0];
  int oc = pads[1] + pads[5] + input_shape[1];
  int oh = pads[2] + pads[6] + input_shape[2];
  int ow = pads[3] + pads[7] + input_shape[3];
  assert(on == shape[0]);
  assert(oc == shape[1]);
  assert(oh == shape[2]);
  assert(ow == shape[3]);
  pad_constant(input_data->data(), output_data->data(), input_shape, pads,
               const_val);
}
void PadOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir