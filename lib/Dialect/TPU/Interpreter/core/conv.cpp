#include "tpuc/Interpreter/cpu/conv.hpp"
#include "internal.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/pad.hpp"
#include "tpuc/MlirModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"

namespace mlir {


Conv2DOpKernel::Conv2DOpKernel(Operation &op, value_map_t &valueMapping,
                               weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto castOp = cast<tpu::Conv2DOp>(op);
  parseConvParam(castOp.param(), is_deconv, castOp.input(), castOp.output(),
                 n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h,
                 ins_w, sh, sw, pt, pb, pl, pr, dh, dw, is_dw, with_bias,
                 do_relu, pad_value);
  this->quant_scale = this->opdTensors[3];
  this->quant_zeropoint = this->opdTensors[4];
  this->quant_rshift = this->opdTensors[5];
  this->quant_multiplier = this->opdTensors[6];

  auto input_type = castOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();

  auto filter_type = castOp.filter().getType().template cast<TensorType>();
  this->filter_shape = filter_type.getShape();

  // get tensors
  assert(this->opdTensors.size() == 7);
  input_data = this->opdTensors[0];
  filter_data = this->opdTensors[1];
  bias_data = this->opdTensors[2];
  output_data = this->resTensor;
  zero_bias = std::make_shared<TensorData>(oc, 0.0f);
  if (datatype == DataType::BF16) {
    if (getOpQuantParamType(&op) == "ACTIVATION_BF16") {
      activation_bf16 = true;
      int quant_size = quant_scale->size();
      int inner_size = (ic / g) * kh * kw;
      int filter_size = filter_data->size();
      assert(filter_size == inner_size * quant_size);
      for(int i = 0;i < filter_size; i++) {
        int q_idx = i / inner_size;
        auto scale = quant_scale->at(q_idx);
        auto zeropoint = quant_zeropoint->at(q_idx);
        filter_data->at(i) = BF16(BF16(filter_data->at(i) * scale) + zeropoint);
      }
    }
  }

  // set mkldnn
  // in int8 case, bias will be add after mkldnn conv
  // reason is int8 case, bias format is 32bit
  float *bias = nullptr;
  if (!with_bias) {
    bias_data = zero_bias;
    bias = zero_bias->data();
  } else if (datatype == DataType::INT8) {
    bias = zero_bias->data();
  } else {
    bias = bias_data->data();
  }
  if (ins_w || ins_h) {
    int ih_after = (ih - 1) * (ins_h + 1) + 1;
    int iw_after = (iw - 1) * (ins_w + 1) + 1;
    tmp_input_data.resize(n * ic * ih_after * iw_after);
    conv.setup(filter_data->data(), bias,
              n, ic, ih_after, iw_after, oc, oh, ow, kh, kw, sh, sw, dh, dw,
              pt, pb, pl, pr, g);
  } else {
    conv.setup(filter_data->data(), bias,
              n, ic, ih, iw, oc, oh, ow, kh, kw, sh, sw, dh, dw,
              pt, pb, pl, pr, g);
  }
}

void Conv2DOpKernel::invoke() {
  if (ins_w || ins_h) {
    my_dilateActivation(input_data->data(), tmp_input_data.data(),
                        0, 0, ins_h, 0, 0, 0, ins_w, 0, n,
                        ic, ih, iw, 0);
    conv.run(tmp_input_data.data(), output_data->data());
  } else {
    conv.run(input_data->data(), output_data->data());
  }

  if (this->datatype == DataType::FP32) {
    if (do_relu) {
      relu(output_data->data(), output_data->data(), output_data->size());
    }
  } else if (this->datatype == DataType::BF16) {
    if (do_relu) {
      relu(output_data->data(), output_data->data(), output_data->size());
    }
    BF16(output_data->data(), output_data->data(), output_data->size());
  } else {
    quantizeActivationInt8PerChannelMultiplierAndRShift(
            output_data->data(), output_data->data(), bias_data->data(), do_relu,
            n, oc, oh * ow, quant_rshift->data(), quant_multiplier->data());
  }
}

} // namespace mlir
