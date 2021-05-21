#include "tpuc/Interpreter/cpu/deconv.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/activation.hpp"
#include "tpuc/ModuleInterpreter.h"
namespace mlir {

DeConv2DOpKernel::DeConv2DOpKernel(Operation &op, value_map_t &valueMapping)
  : CPUOpKernel(op, valueMapping) {
  auto castOp = cast<tpu::DeConv2DOp>(op);
  parseConvParam(castOp.param(), is_deconv, castOp.input(), castOp.output(),
                 castOp.filter(), n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw,
                 pt, pb, pl, pr, dh, dw, is_dw, with_bias, do_relu, pad_value);
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

  // set mkldnn
  this->mkl_eng = mkldnn::engine(mkldnn::engine::kind::cpu, 0);
  this->mkl_stream = mkldnn::stream(mkl_eng);

  mkldnn::memory::dims mkl_src_shape = {n, ic, ih, iw};
  mkldnn::memory::dims mkl_filter_shape =
      (g != 1) ? mkldnn::memory::dims{g, oc / g, ic / g, kh, kw}
               : mkldnn::memory::dims{oc, ic, kh, kw};
  mkldnn::memory::dims mkl_bias_shape = {oc};
  mkldnn::memory::dims mkl_dst_shape = {n, oc, oh, ow};
  mkldnn::memory::dims mkl_strides = {sh, sw};

  mkldnn::memory::dims mkl_padding_l = {pt, pl};
  mkldnn::memory::dims mkl_padding_r = {pb, pr};
  mkldnn::memory::dims mkl_dilation = {
      dh - 1, dw - 1}; // mkldnn dialtion is different with caffe

  using tag = mkldnn::memory::format_tag;
  using dt = mkldnn::memory::data_type;

  // set mkldnn memory
  mkldnn::memory mkl_src_memory = mkldnn::memory(
      {{mkl_src_shape}, dt::f32, tag::nchw}, mkl_eng, input_data->data());

  mkldnn::memory mkl_filter_memory =
      (g != 1) ? mkldnn::memory({{mkl_filter_shape}, dt::f32, tag::goihw},
                                mkl_eng, filter_data->data())
               : mkldnn::memory({{mkl_filter_shape}, dt::f32, tag::oihw},
                                mkl_eng, filter_data->data());

  zero_bias = std::make_shared<std::vector<float>>(oc, 0.0f);

  // in int8 case, bias will be add after mkldnn conv
  // reason is int8 case, bias format is 32bit
  bool do_bias = with_bias;
  if (!do_bias) {
    bias_data = zero_bias;
  }
  if (use_multiplier) {
    do_bias = false;
  }

  mkldnn::memory mkl_bias_memory =
      mkldnn::memory({{mkl_bias_shape}, dt::f32, tag::x}, mkl_eng,
                     do_bias ? bias_data->data() : zero_bias->data());
  mkldnn::memory mkl_dst_memory = mkldnn::memory(
      {{mkl_dst_shape}, dt::f32, tag::nchw}, mkl_eng, output_data->data());

  mkldnn::memory::desc src_md =
      mkldnn::memory::desc({mkl_src_shape}, dt::f32, tag::any);

  mkldnn::memory::desc filter_md =
      mkldnn::memory::desc({mkl_filter_shape}, dt::f32, tag::any);
  mkldnn::memory::desc bias_md =
      mkldnn::memory::desc({mkl_bias_shape}, dt::f32, tag::any);
  mkldnn::memory::desc dst_md =
      mkldnn::memory::desc({mkl_dst_shape}, dt::f32, tag::any);

  mkldnn::deconvolution_forward::desc deconv_desc =
      mkldnn::deconvolution_forward::desc(
          mkldnn::prop_kind::forward_inference,
          mkldnn::algorithm::deconvolution_direct, src_md, filter_md, bias_md,
          dst_md, mkl_strides, mkl_dilation, mkl_padding_l, mkl_padding_r);
  mkldnn::deconvolution_forward::primitive_desc deconv_prim_desc =
      mkldnn::deconvolution_forward::primitive_desc(deconv_desc, mkl_eng);

  // do reorder if needed
  auto prim_src_memory = mkl_src_memory;
  if (deconv_prim_desc.src_desc() != mkl_src_memory.get_desc()) {
    prim_src_memory = mkldnn::memory(deconv_prim_desc.src_desc(), mkl_eng);
    mkl_net.push_back(mkldnn::reorder(mkl_src_memory, prim_src_memory));
    mkl_net_args.push_back(
        {{MKLDNN_ARG_FROM, mkl_src_memory}, {MKLDNN_ARG_TO, prim_src_memory}});
  }
  auto prim_filter_memory = mkl_filter_memory;
  if (deconv_prim_desc.weights_desc() != mkl_filter_memory.get_desc()) {
    prim_filter_memory =
        mkldnn::memory(deconv_prim_desc.weights_desc(), mkl_eng);
    mkldnn::reorder(mkl_filter_memory, prim_filter_memory)
        .execute(mkl_stream, mkl_filter_memory, prim_filter_memory);
  }
  auto prim_dst_memory = mkldnn::memory(deconv_prim_desc.dst_desc(), mkl_eng);

  mkl_net.push_back(mkldnn::deconvolution_forward(deconv_prim_desc));
  mkl_net_args.push_back({{MKLDNN_ARG_SRC, prim_src_memory},
                          {MKLDNN_ARG_WEIGHTS, prim_filter_memory},
                          {MKLDNN_ARG_BIAS, mkl_bias_memory},
                          {MKLDNN_ARG_DST, prim_dst_memory}});

  // reorder or copy the output
  if (prim_dst_memory != mkl_dst_memory) {
    mkl_net.push_back(mkldnn::reorder(prim_dst_memory, mkl_dst_memory));
    mkl_net_args.push_back(
        {{MKLDNN_ARG_FROM, prim_dst_memory}, {MKLDNN_ARG_TO, mkl_dst_memory}});
  }
  assert(mkl_net.size() == mkl_net_args.size() && "something is missing");
}

void DeConv2DOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " DeConv op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};
std::vector<float> DeConv2DOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void DeConv2DOpKernel::fp32_invoke() {
  for (size_t i = 0; i < mkl_net.size(); ++i) {
    mkl_net.at(i).execute(mkl_stream, mkl_net_args.at(i));
  }
  mkl_stream.wait();
  if (do_relu) {
    relu(output_data->data(), output_data->data(), output_data->size());
  }
};

void DeConv2DOpKernel::i8_invoke() {
  for (size_t i = 0; i < mkl_net.size(); ++i) {
    mkl_net.at(i).execute(mkl_stream, mkl_net_args.at(i));
  }
  mkl_stream.wait();
  if (is_perchannel) {
    if (use_multiplier) {
      quantizeActivationInt8PerChannelMultiplierAndRShift(
          output_data->data(), output_data->data(), bias_data->data(), do_relu,
          n, oc, oh * ow, rshift.data(), multiplier.data());
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
};

void DeConv2DOpKernel::invoke() {
  if (this->datatype == DataType::FP32) {
    fp32_invoke();
  } else if (this->datatype == DataType::INT8) {
    i8_invoke();
  } else {
    fp32_invoke();
    clean16bitmantissa(output_data->data(), output_data->data(),
                       output_data->size());
  }
}

void DeConv2DOpKernel::dump() {
  OpKernel::dump();
  std::string filter_shape_str, input_shape_str;
  for (auto &i : this->input_shape) {
    input_shape_str = input_shape_str + std::to_string(i) + " ";
  }
  for (auto &i : this->filter_shape) {
    filter_shape_str = filter_shape_str + std::to_string(i)  + " ";
  }

  llvm::outs() << "\tInput Shape: " << input_shape_str << "\n";
  llvm::outs() << "\tFilter Shape: " << filter_shape_str << "\n";
  llvm::outs() << "\tDo_RELU: " << do_relu << "\n";
  if (this->datatype == DataType::INT8) {
    llvm::outs() << "\tPERCHANNEL: " << is_perchannel << "\n";
    llvm::outs() << "\tMULTIPLIER: " << use_multiplier << "\n";
  }
}
} // namespace mlir