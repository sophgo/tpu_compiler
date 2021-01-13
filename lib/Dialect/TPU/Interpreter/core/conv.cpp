#include "tpuc/Interpreter/cpu/conv.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {
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
  if (!with_bias) {
    auto zero_bias = std::make_shared<std::vector<float>>(oc, 0.0f);
    bias_data = zero_bias;
  }
  mkldnn::memory mkl_bias_memory = mkldnn::memory(
      {{mkl_bias_shape}, dt::f32, tag::x}, mkl_eng, bias_data->data());
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

void Conv2DOpKernel::invoke() {
  for (size_t i = 0; i < mkl_net.size(); ++i) {
    mkl_net.at(i).execute(mkl_stream, mkl_net_args.at(i));
  }
  mkl_stream.wait();
}

void Conv2DOpKernel::dump() {
  std::string shape_str, input_shape_str, filter_shape_str;
  for (auto &i : this->shape) {
    shape_str = shape_str + std::to_string(i) + " ";
  }
  for (auto &i : this->input_shape) {
    input_shape_str = input_shape_str + std::to_string(i) + " ";
  }
  for (auto &i : this->filter_shape) {
    filter_shape_str = filter_shape_str + std::to_string(i) + " ";
  }
  llvm::outs() << "Conv Op\n";
  llvm::outs() << "\tName: " << this->name << "\n";
  llvm::outs() << "\tInput Shape: " << input_shape_str << "\n";
  llvm::outs() << "\tShape: " << shape_str << "\n";
  llvm::outs() << "\tFilter Shape: " << filter_shape_str << "\n";
}
} // namespace mlir