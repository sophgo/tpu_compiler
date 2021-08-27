#include "tpuc/Interpreter/cpu/deconv.hpp"
#include "internal.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/MlirModuleInterpreter.h"

namespace mlir {

DeConv2DOpKernel::DeConv2DOpKernel(Operation &op, value_map_t &valueMapping,
                                   weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto castOp = cast<tpu::DeConv2DOp>(op);
  int no_use0, no_use1;
  parseConvParam(castOp.param(), is_deconv, castOp.input(), castOp.output(),
                 castOp.filter(), n, ic, ih, iw, oc, oh, ow, g, kh, kw, no_use0,
                 no_use1, sh, sw, pt, pb, pl, pr, dh, dw, is_dw, with_bias,
                 do_relu, pad_value);
  is_asymmetric = isOpQuantAsymmetric(&op);
  // int8 init
  if (datatype == DataType::INT8) {
    auto quant_rshift = this->opdTensors[5];
    auto quant_multiplier = this->opdTensors[6];
    assert(quant_rshift);
    if (!isOpQuantPerchannel(&op)) {
      this->is_perchannel = false;
      this->rshift.push_back(quant_rshift->at(0));
    } else {
      this->is_perchannel = true;
      this->rshift.assign(quant_rshift->begin(), quant_rshift->end());
      if (getOpQuantParamType(&op) == "RSHIFT_AND_M_I32") {
        assert(quant_multiplier);
        this->use_multiplier = true;
        this->multiplier.assign(quant_multiplier->begin(),
                                quant_multiplier->end());
      }
    }
  }
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

  // in int8 case, bias will be add after mkldnn conv
  // reason is int8 case, bias format is 32bit
  float *bias = nullptr;
  if (!with_bias) {
    bias_data = zero_bias;
    bias = zero_bias->data();
  } else if (use_multiplier) {
    bias = zero_bias->data();
  } else {
    bias = bias_data->data();
  }
  deconv.setup(filter_data->data(), bias, n, ic, ih, iw, oc, oh, ow, kh, kw, sh,
               sw, dh, dw, pt, pb, pl, pr, g);
}

void DeConv2DOpKernel::invoke() {
  deconv.run(input_data->data(), output_data->data());
  if (this->datatype == DataType::FP32) {
    if (do_relu) {
      relu(output_data->data(), output_data->data(), output_data->size());
    }
  } else if (this->datatype == DataType::BF16) {
    if (do_relu) {
      relu(output_data->data(), output_data->data(), output_data->size());
    }
    BF16(output_data->data(), output_data->data(), output_data->size(), true);
  } else {
    if (is_perchannel) {
      if (use_multiplier) {
        quantizeActivationInt8PerChannelMultiplierAndRShift(
            output_data->data(), output_data->data(), bias_data->data(),
            do_relu, n, oc, oh * ow, rshift.data(), multiplier.data());
      } else {
        if (do_relu) {
          relu(output_data->data(), output_data->data(), output_data->size());
        }
        quantizeActivationInt8PerChannelRShift(output_data->data(),
                                               output_data->data(), n, oc,
                                               oh * ow, rshift.data());
      }
    } else {
      if (do_relu) {
        relu(output_data->data(), output_data->data(), output_data->size());
      }
      quantizeActivationInt8PerLayerRshift(
          output_data->data(), output_data->data(), size, rshift.at(0));
    }
  }
}

} // namespace mlir