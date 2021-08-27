#include "tpuc/Interpreter/cpu/depthtospace.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/MlirModuleInterpreter.h"


void pixelshuffle(float *input, float *output, int in, int ic, int ih, int iw,
                  int on, int oc, int oh, int ow, int upscale_factor,
                  bool dcr_mode) {
  int i_index = 0, o_index = 0, new_c = 0, new_h = 0, new_w = 0,
      r = upscale_factor;
  LLVM_DEBUG(llvm::errs() << "  in: " << in << ", ic: " << ic << ", ih: " << ih
                          << ", iw: " << iw << "\n";);
  LLVM_DEBUG(llvm::errs() << "  on: " << on << ", oc: " << oc << ", oh: " << oh
                          << ", ow: " << ow << "\n";);
  if (dcr_mode) {
    for (int n = 0; n < in; n++) {
      for (int c = 0; c < ic; c++) {
        for (int h = 0; h < ih; h++) {
          for (int w = 0; w < iw; w++) {
            new_c = c % oc;
            new_h = h * r + static_cast<int>(floor((c / oc) / r));
            new_w = w * r + ((c / oc) % r);
            o_index =
                n * (oc * oh * ow) + new_c * (oh * ow) + new_h * ow + new_w;
            output[o_index] = input[i_index];
            i_index++;
          }
        }
      }
    }
  } else {
    for (int n = 0; n < in; n++) {
      for (int c = 0; c < ic; c++) {
        for (int h = 0; h < ih; h++) {
          for (int w = 0; w < iw; w++) {
            new_c = static_cast<int>(floor(c / (r * r)));
            new_h = h * r + (static_cast<int>(floor(c / r))) % r;
            new_w = w * r + (c % (r * r)) % r;
            o_index =
                n * (oc * oh * ow) + new_c * (oh * ow) + new_h * ow + new_w;
            output[o_index] = input[i_index];
            i_index++;
          }
        }
      }
    }
  }
}

namespace mlir {

DepthToSpaceOpKernel::DepthToSpaceOpKernel(Operation &op,
                                           value_map_t &valueMapping,
                                           weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto dtsOp = cast<tpu::PixelShuffleOp>(op);
  auto input_type = dtsOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();
  this->upscale_factor = dtsOp.upscale_factor();
  this->dcr_mode = dtsOp.mode() == "DCR";
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void DepthToSpaceOpKernel::invoke() {
  if (shape.size() < 4 || input_shape.size() < 4) {
    dump();
    llvm_unreachable("wrong shape size");
  }
  int on = shape[0];
  int oc = shape[1];
  int oh = shape[2];
  int ow = shape[3];
  int ic = input_shape[1];
  int ih = input_shape[2];
  int iw = input_shape[3];
  pixelshuffle(input_data->data(), output_data->data(), on, ic, ih, iw, on, oc,
               oh, ow, upscale_factor, dcr_mode);
}

} // namespace mlir