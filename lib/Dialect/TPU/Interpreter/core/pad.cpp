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
  // when pads < 0 means cutoff
  int32_t start_in = pads[0] < 0 ? -pads[0] : 0;
  int32_t start_ic = pads[1] < 0 ? -pads[1] : 0;
  int32_t start_ih = pads[2] < 0 ? -pads[2] : 0;
  int32_t start_iw = pads[3] < 0 ? -pads[3] : 0;

  int32_t end_in = pads[4] < 0 ? in + pads[4]: in;
  int32_t end_ic = pads[5] < 0 ? ic + pads[5]: ic;
  int32_t end_ih = pads[6] < 0 ? ih + pads[6]: ih;
  int32_t end_iw = pads[7] < 0 ? iw + pads[7]: iw;

  int32_t pad_n_begin_size = pads[0] < 0 ? 0 : pads[0] * oc * oh * ow;
  int32_t pad_c_begin_size = pads[1] < 0 ? 0 : pads[1] * oh * ow;
  int32_t pad_h_begin_size = pads[2] < 0 ? 0 : pads[2] * ow;
  int32_t pad_w_begin_size = pads[3] < 0 ? 0 : pads[3];

  for (int out_idx = 0, in_idx = start_in; in_idx < end_in; in_idx++, out_idx++) {
    auto in_offset = in_idx * ic * ih * iw;
    auto out_offset = pad_n_begin_size + pad_c_begin_size + out_idx * oc * oh * ow;
    for (int oc_idx = 0, ic_idx = start_ic; ic_idx < end_ic; ic_idx++, oc_idx++) {
      auto in_ic_offset = in_offset + ic_idx * ih * iw;
      auto out_oc_offset = out_offset + pad_h_begin_size + oc_idx * oh * ow;
      for (int oh_idx = 0, ih_idx = start_ih; ih_idx < end_ih; ih_idx++, oh_idx++) {
        auto in_ih_offset = in_ic_offset + ih_idx * iw;
        auto out_oh_offset = out_oc_offset + pad_w_begin_size + oh_idx * ow;
        memcpy(output + out_oh_offset, input + in_ih_offset + start_iw ,
               (end_iw - start_iw) * sizeof(float_t));
      } // end h
    } // end c
  } // end n
} // end func

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
