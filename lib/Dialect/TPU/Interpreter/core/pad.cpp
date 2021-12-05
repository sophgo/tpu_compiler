#include "tpuc/Interpreter/cpu/pad.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/MlirModuleInterpreter.h"

namespace mlir {

void pad_edge(float *input, float *output, std::vector<int64_t> &input_shape,
              std::vector<int> &pads) {
  int in = input_shape[0];
  int ic = input_shape[1];
  int ih = input_shape[2];
  int iw = input_shape[3];
  // int on = pads[0] + pads[4] + in;
  int oc = pads[1] + pads[5] + ic;
  int oh = pads[2] + pads[6] + ih;
  int ow = pads[3] + pads[7] + iw;
  assert(pads[0] == pads[4] && pads[1] == pads[5] && pads[0] == 0 &&
         pads[1] == 0 && "only support hw pad");

  // comes from https://github.com/BVLC/caffe/pull/6506/files
  for (int n = 0; n < in; ++n) {
    for (int c = 0; c < ic; ++c) {
      // First copy the main body into place
      for (int h = 0; h < ih; ++h) {
        // copy the width part
        int input_offset = ((n * ic + c) * ih + h) * iw;
        int output_offset = ((n * oc + c) * oh + (h + pads[2])) * ow + pads[3];

        memcpy(output + output_offset, input + input_offset,
               sizeof(float) * iw);
      }

      // Left and right. Loop over the rows not in the vertical padding
      for (int h = pads[2]; h < oh - pads[6]; ++h) {
        // Offset to current row start (in padding of this row)
        int off = ((n * oc + c) * oh + h) * ow;
        const float lval = *(output + off + pads[3]),
                    rval = *(output + off + ow - 1 - pads[7]);

        // Left
        for (int wdst = 0; wdst < pads[3]; ++wdst) {
          *(output + off + wdst) = lval;
        }
        // Right
        for (int wdst = ow - pads[7]; wdst < ow; ++wdst) {
          *(output + off + wdst) = rval;
        }
      }

      // Top
      // Beginning of this image's data, including padding
      float *dstptr = output + ((n * oc + c) * oh) * ow;
      // First row not in the vertical padding
      float *srcptr = dstptr + pads[2] * ow;
      for (int h = 0; h < pads[2]; ++h) {
        std::copy(srcptr, srcptr + ow, dstptr + h * ow);
      }

      // Bottom
      // Start of last row not in the vertical padding
      srcptr = output + ((n * oc + c) * oh + (oh - 1 - pads[6])) * ow;
      // Start of first row in bottom padding
      dstptr = srcptr + ow;
      for (int h = 0; h < pads[6]; ++h) {
        std::copy(srcptr, srcptr + ow, dstptr + h * ow);
      }
    }
  }
}

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

PadOpKernel::PadOpKernel(Operation &op, value_map_t &valueMapping,
                         weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto padOp = cast<tpu::PadOp>(op);
  parsePadParam<tpu::PadOp>(&op, input_shape, output_shape, pads);
  this->const_val = padOp.const_val().convertToFloat();
  this->mode = padOp.mode().str();
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void PadOpKernel::invoke() {
  int on = pads[0] + pads[4] + input_shape[0];
  int oc = pads[1] + pads[5] + input_shape[1];
  int oh = pads[2] + pads[6] + input_shape[2];
  int ow = pads[3] + pads[7] + input_shape[3];
  assert(on == output_shape[0]);
  assert(oc == output_shape[1]);
  assert(oh == output_shape[2]);
  assert(ow == output_shape[3]);
  if (this->mode == "edge") {
    pad_edge(input_data->data(), output_data->data(), input_shape, pads);
  } else {
    pad_constant(input_data->data(), output_data->data(), input_shape, pads,
                 const_val);
  }
}

} // namespace mlir
