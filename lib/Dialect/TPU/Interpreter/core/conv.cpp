#include "tpuc/Interpreter/cpu/conv.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/activation.hpp"
#include "tpuc/Interpreter/cpu/pad.hpp"
#include "tpuc/ModuleInterpreter.h"
namespace mlir {

void once_mkldnn_conv(float *input, float *weight, float *bias, float *output,
                      int n, int ic, int ih, int iw, int oc, int oh, int ow,
                      int kh, int kw, int sh, int sw, int dh, int dw, int pt,
                      int pb, int pl, int pr, int g, int pad_value) {
  std::shared_ptr<std::vector<float>> zero_bias = nullptr;
  if (!bias) {
    zero_bias = std::make_shared<std::vector<float>>(oc, 0.0f);
    bias = zero_bias->data();
  }

  LLVM_DEBUG(llvm::errs() << "  k: (" << kh << "*" << kw << "), "
                          << "s: (" << sh << "*" << sw << "), "
                          << "pt: " << pt << " pb: " << pb << " pl: " << pl
                          << " pr: " << pr << " g: " << g << "\n";
             llvm::errs() << "n:" << n << " c: " << ic << " h:" << ih
                          << " w:" << iw << "\n"
                          << " oc: " << oc << " oh:" << oh << " ow:" << ow
                          << "\n";
             llvm::errs() << "pad value: " << pad_value << "\n";);
  using namespace mkldnn;
  using tag = memory::format_tag;
  using dt = memory::data_type;

  engine eng(engine::kind::cpu, 0);
  stream s(eng);

  std::vector<primitive> net;
  std::vector<std::unordered_map<int, memory>> net_args;

  std::vector<float> input_after_pad;
  // mkldnn not support non zero padding
  // we handle it.
  if (pad_value != 0) {
    if (pt != 0 || pl != 0 || pb != 0 || pr != 0) {
      input_after_pad.resize(n * ic * (ih + pt + pb) * (iw + pl + pr));
      std::vector<int> pads = {0, 0, pt, pl, 0, 0, pb, pr};
      std::vector<int64_t> input_shape = {n, ic, ih, iw};
      pad_constant(input, input_after_pad.data(), input_shape, pads, pad_value);
      input = input_after_pad.data();
      ih = ih + pt + pb;
      iw = iw + pl + pr;
      pt = 0;
      pb = 0;
      pl = 0;
      pr = 0;
    }
  }

  const memory::dim batch = n;
  memory::dims src_tz = {batch, ic, ih, iw};
  memory::dims weights_tz = (g != 1) ? memory::dims{g, oc / g, ic / g, kh, kw}
                                     : memory::dims{oc, ic, kh, kw};
  memory::dims bias_tz = {oc};
  memory::dims dst_tz = {batch, oc, oh, ow};
  memory::dims strides = {sh, sw};

  memory::dims padding_l = {pt, pl};
  memory::dims padding_r = {pb, pr};
  memory::dims dilation = {dh - 1,
                           dw - 1}; // mkldnn dialtion is different with caffe

  // memory
  auto user_src_memory = memory({{src_tz}, dt::f32, tag::nchw}, eng, input);
  auto user_weights_memory =
      (g != 1) ? memory({{weights_tz}, dt::f32, tag::goihw}, eng, weight)
               : memory({{weights_tz}, dt::f32, tag::oihw}, eng, weight);
  auto user_bias_memory = memory({{bias_tz}, dt::f32, tag::x}, eng, bias);
  auto user_dst_memory = memory({{dst_tz}, dt::f32, tag::nchw}, eng, output);

  // md
  auto src_md = memory::desc({src_tz}, dt::f32, tag::any);
  auto weights_md = memory::desc({weights_tz}, dt::f32, tag::any);
  auto bias_md = memory::desc({bias_tz}, dt::f32, tag::any);
  auto dst_md = memory::desc({dst_tz}, dt::f32, tag::any);

  // conv desc
  auto conv_desc = convolution_forward::desc(
      prop_kind::forward_inference, algorithm::convolution_direct, src_md,
      weights_md, bias_md, dst_md, strides, dilation, padding_l, padding_r);
  auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, eng);

  // do reorder if needed
  auto src_memory = user_src_memory;
  if (conv_prim_desc.src_desc() != user_src_memory.get_desc()) {
    src_memory = memory(conv_prim_desc.src_desc(), eng);
    net.push_back(reorder(user_src_memory, src_memory));
    net_args.push_back(
        {{MKLDNN_ARG_FROM, user_src_memory}, {MKLDNN_ARG_TO, src_memory}});
  }
  auto weights_memory = user_weights_memory;
  if (conv_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
    weights_memory = memory(conv_prim_desc.weights_desc(), eng);
    reorder(user_weights_memory, weights_memory)
        .execute(s, user_weights_memory, weights_memory);
  }
  auto bias_memory = user_bias_memory;

  auto dst_memory = memory(conv_prim_desc.dst_desc(), eng);

  net.push_back(convolution_forward(conv_prim_desc));
  net_args.push_back({{MKLDNN_ARG_SRC, src_memory},
                      {MKLDNN_ARG_WEIGHTS, weights_memory},
                      {MKLDNN_ARG_BIAS, bias_memory},
                      {MKLDNN_ARG_DST, dst_memory}});

  // reorder or copy the output
  if (dst_memory != user_dst_memory) {
    net.push_back(reorder(dst_memory, user_dst_memory));
    net_args.push_back(
        {{MKLDNN_ARG_FROM, dst_memory}, {MKLDNN_ARG_TO, user_dst_memory}});
  }

  // run
  assert(net.size() == net_args.size() && "something is missing");
  for (size_t i = 0; i < net.size(); ++i)
    net.at(i).execute(s, net_args.at(i));

  s.wait();
}

Conv2DOpKernel::Conv2DOpKernel(Operation &op, value_map_t &valueMapping) {
  auto castOp = cast<tpu::Conv2DOp>(op);
  assert(castOp);
  llvm::outs() << " Conv op: [" << castOp.name() << "]\n";

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = castOp.getResult();
  auto size = getTensorSize(result);
  llvm::outs() << "    =>required memory size: [" << size << "]\n";

  auto resultTensor = std::make_shared<std::vector<float>>(size);
  parseConvParam(castOp.param(), is_deconv, castOp.input(), castOp.output(),
                 castOp.filter(), n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw,
                 pt, pb, pl, pr, dh, dw, is_dw, with_bias, do_relu, pad_value);
  is_asymmetric = isOpQuantAsymmetric(&op);
  this->name = castOp.name().str();
  this->op_type = op.getName().getStringRef().str();

  arrayAttrToVector(castOp.param().ins(), ins);

  set_datatype(getOpQuant(&op).str());

  // int8 init
  if (datatype == DataType::INT8) {
    auto quant_rshift = opTensors[5];
    auto quant_multiplier = opTensors[6];
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

  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  auto input_type = castOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();

  auto filter_type = castOp.filter().getType().template cast<TensorType>();
  this->filter_shape = filter_type.getShape();

  // get tensors
  assert(opTensors.size() == 7);
  input_data = opTensors[0];
  filter_data = opTensors[1];
  bias_data = opTensors[2];

  output_data = resultTensor;

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

  mkldnn::convolution_forward::desc conv_desc =
      mkldnn::convolution_forward::desc(
          mkldnn::prop_kind::forward_inference,
          mkldnn::algorithm::convolution_direct, src_md, filter_md, bias_md,
          dst_md, mkl_strides, mkl_dilation, mkl_padding_l, mkl_padding_r);
  mkldnn::convolution_forward::primitive_desc conv_prim_desc =
      mkldnn::convolution_forward::primitive_desc(conv_desc, mkl_eng);

  // do reorder if needed
  auto prim_src_memory = mkl_src_memory;
  if (conv_prim_desc.src_desc() != mkl_src_memory.get_desc()) {
    prim_src_memory = mkldnn::memory(conv_prim_desc.src_desc(), mkl_eng);
    mkl_net.push_back(mkldnn::reorder(mkl_src_memory, prim_src_memory));
    mkl_net_args.push_back(
        {{MKLDNN_ARG_FROM, mkl_src_memory}, {MKLDNN_ARG_TO, prim_src_memory}});
  }
  auto prim_filter_memory = mkl_filter_memory;
  if (conv_prim_desc.weights_desc() != mkl_filter_memory.get_desc()) {
    prim_filter_memory = mkldnn::memory(conv_prim_desc.weights_desc(), mkl_eng);
    mkldnn::reorder(mkl_filter_memory, prim_filter_memory)
        .execute(mkl_stream, mkl_filter_memory, prim_filter_memory);
  }
  auto prim_dst_memory = mkldnn::memory(conv_prim_desc.dst_desc(), mkl_eng);

  mkl_net.push_back(mkldnn::convolution_forward(conv_prim_desc));
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

  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void Conv2DOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " Conv op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};
std::vector<float> Conv2DOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void Conv2DOpKernel::fp32_invoke() {
  for (size_t i = 0; i < mkl_net.size(); ++i) {
    mkl_net.at(i).execute(mkl_stream, mkl_net_args.at(i));
  }
  mkl_stream.wait();
  if (do_relu) {
    relu(output_data->data(), output_data->data(), output_data->size());
  }
};

void Conv2DOpKernel::i8_invoke() {
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

void Conv2DOpKernel::invoke() {
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

void Conv2DOpKernel::dump() {
  OpKernel::dump();
  std::string filter_shape_str, input_shape_str;
  for (auto &i : this->input_shape) {
    input_shape_str = input_shape_str + std::to_string(i) + " ";
  }
  for (auto &i : this->filter_shape) {
    filter_shape_str = filter_shape_str + std::to_string(i) + " ";
  }

  llvm::outs() << "\tInput Shape: " << input_shape_str << "\n";
  llvm::outs() << "\tFilter Shape: " << filter_shape_str << "\n";
  llvm::outs() << "\tPad top: " << pt << " bottom: " << pb << " left: " << pl
               << " right: " << pr << "\n";
  llvm::outs() << "\tDo_RELU: " << do_relu << "\n";
  if (this->datatype == DataType::INT8) {
    llvm::outs() << "\tPERCHANNEL: " << is_perchannel << "\n";
    llvm::outs() << "\tMULTIPLIER: " << use_multiplier << "\n";
  }
}
} // namespace mlir