#include "tpuc/Interpreter/cpu/conv3d.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/activation.hpp"
#include "tpuc/Interpreter/cpu/pad.hpp"
#include "tpuc/ModuleInterpreter.h"
namespace mlir {

static void get_strides_from_shapes5d(
    int strides[5], const int shapes[5], int ws)
{
  strides[5 - 1] = ws;
  for (int i = 5 - 2; i >= 0; i--)
    strides[i] = shapes[i + 1] * strides[i + 1];
}

static int get_tensor5d_offset(
    int poss[5], const int strides[5])
{
  int offset = 0;
  for (int i = 0; i < 5; i++)
    offset += poss[i] * strides[i];

  return offset;
}

static inline void ins_and_pad(float* after, float *before,
   int in_h, int in_w, int pad_t, int pad_l, int ins_w, int ins_h,
   int w_after, int h_after)
{
  for (int h = 0; h < in_h; h++) {
    for (int w = 0; w < in_w; w++) {
      int i = (h * (ins_h + 1) + pad_t) * w_after + w * (ins_w + 1) + pad_l;
      after[i] = before[h * in_w + w];
    }
  }
}

// input (n, ic, id, ih, iw)
// output (n, oc, od, oh, ow)
// weight (oc, ic, kd, kh, kw), pytorch
void _conv3d_float_ref(float *_input, float *weight, float *bias, float *output,
  int batch, int input_c, int input_d, int input_h, int input_w,
  int output_c, int output_d, int output_h, int output_w,
  int kernel_d, int kernel_h, int kernel_w,
  int stride_d, int stride_h, int stride_w,
  int dilation_d, int dilation_h, int dilation_w,
  int pad_d0, int pad_top, int pad_bottom,
  int pad_d1, int pad_left, int pad_right) {

  (void)pad_d1;
  (void)pad_bottom;
  (void)pad_right;

  // expand pad: hw->expand d
  // TODO: not pad it
  // TODO: check dilate > 1 case
  float* input = _input;
  int input_h_pad = input_h + pad_top + pad_bottom;
  int input_w_pad = input_w + pad_left + pad_right;
  int input_d_pad = input_d + pad_d0 + pad_d1;
  int new_input_sz = batch * input_c * input_d_pad * input_h_pad * input_w_pad;

  // new buffer
  auto _input_pad = std::vector<float>(new_input_sz, 0);

  int input_shapes[5] = {batch, input_c, input_d, input_h, input_w};
  int output_shapes[5] = {batch, output_c, output_d, output_h, output_w};

  //int kernel_shapes[5] = {output_c, kernel_d, kernel_h, kernel_w, input_c};
  int kernel_shapes[5] = {output_c, input_c, kernel_d, kernel_h, kernel_w};

  int input_strides[5];
  int output_strides[5];
  int kernel_strides[5];

  // input/output shape (n, c, d, h, w)
  get_strides_from_shapes5d(input_strides, input_shapes, sizeof(float));

  if (1) {
    assert(dilation_d == 1 && "not support dilation_d > 1");
    assert(dilation_h == 1 && "not support dilation_h > 1");
    assert(dilation_w == 1 && "not support dilation_w > 1");

    int hw_pad = input_h_pad * input_w_pad;
    int dhw_pad = input_d_pad * hw_pad;
    int hw = input_h * input_w;
    int dhw = input_d * hw;
    for (int i = 0; i < batch; ++i) {
      for (int ic = 0; ic < input_c; ++ic) {
        int offset_pad = i * input_c * dhw_pad + ic * dhw_pad + (hw_pad * pad_d0);
        int offset_org = i * input_c * dhw + ic * dhw;
        for (int id = 0; id < input_d; ++id) {
          ins_and_pad(&_input_pad.data()[offset_pad + id * hw_pad],
              &input[offset_org + id * hw],
              input_h, input_w,
              pad_top, pad_left, /*ins_w=*/0, /*ins_h=*/0,
              input_w_pad, input_h_pad);
        }
      }
    }
    input = _input_pad.data();

    // padded, clear it
    pad_d0 =  pad_top =  pad_bottom = 0;
    pad_d1 =  pad_left =  pad_right = 0;
    int input_shapes[5] = {batch, input_c, input_d_pad, input_h_pad, input_w_pad};
    get_strides_from_shapes5d(input_strides, input_shapes, sizeof(float));
  }


  get_strides_from_shapes5d(output_strides, output_shapes, sizeof(float));

  // kernel shape (oc, ic, kd, kh, kw), pytorch
  get_strides_from_shapes5d(kernel_strides, kernel_shapes, sizeof(float));

  for (int i = 0; i < batch; ++i) {
    for (int oc = 0; oc < output_c; oc++) {
      for (int oz = 0; oz < output_d; oz++) {
        for (int oy = 0; oy < output_h; ++oy) {
          for (int ox = 0; ox < output_w; ++ox) {
            for (int ic = 0; ic < input_c; ++ic) {
              for (int kz = 0; kz < kernel_d; ++kz) {
                const int iz = oz * stride_d + kz * dilation_d - pad_d0;
                if (1 || iz < input_d) {
                  for (int ky = 0; ky < kernel_h; ++ky) {
                    const int iy = oy * stride_h + ky * dilation_h - pad_top;
                    if (1 || iy < input_h) {
                      for (int kx = 0; kx < kernel_w; ++kx) {
                        const int ix = ox * stride_w + kx * dilation_w - pad_left;
                        if (1 || ix < input_w) {
                          int input_poss[5] = {i, ic, iz, iy, ix};
                          int input_offset = get_tensor5d_offset(input_poss, input_strides)
                                             / input_strides[5 - 1];

                          // pytorch (oc=1, ic=1, kd=1, kh=3, kw=3)
                          int kernel_poss[5] = {oc, ic, kz, ky, kx};

                          int kernel_offset =
                            get_tensor5d_offset(kernel_poss, kernel_strides)
                                / kernel_strides[5 - 1];

                          int output_poss[5] = {i, oc, oz, oy, ox};
                          int output_offset =
                            get_tensor5d_offset(output_poss, output_strides)
                                / output_strides[5 - 1];

                          output[output_offset] +=
                            input[input_offset] * weight[kernel_offset];
                          //printf("o output[%d](%f) += in[%d](%f) * w[%d](%f)\n",
                          //    output_offset, output[output_offset],
                          //    input_offset, input[input_offset],
                          //    kernel_offset, weight[kernel_offset]);

                        }
                        else {
                          //assert(0 && "w");
                        }
                      }
                    }
                    else {
                      //assert(0 && "h");
                    }
                  }
                }
                else {
                  //assert(0 && "d");
                }
              }
            }
          }
        }
      }
    }
  }

  if (bias) {
    for (int i = 0; i < batch; ++i) {
      for (int oy = 0; oy < output_h; ++oy) {
        for (int ox = 0; ox < output_w; ++ox) {
          for (int oc = 0; oc < output_c; ++oc) {
            for (int od = 0; od < output_d; ++od) {
              int output_poss[5] = {i, oc, od, oy, ox};
              int output_offset =
                  get_tensor5d_offset(output_poss, output_strides)
                      / output_strides[5 - 1];
              output[output_offset] += bias[oc];
            }
          }
        }
      }
    }
  }
}

Conv3DOpKernel::Conv3DOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto castOp = cast<tpu::Conv3DOp>(op);
  parseConv3dParam(castOp.param(), is_deconv,
                  castOp.input(), castOp.output(), castOp.filter(),
                  n, ic, id, ih, iw,
                  oc, od, oh, ow, g,
                  kd, kh, kw,
                  sd, sh, sw,
                  pd0, pd1, pt, pb, pl, pr,
                  dd, dh, dw,
                  is_dw, with_bias, do_relu);

  is_asymmetric = isOpQuantAsymmetric(&op);
  arrayAttrToVector(castOp.param().ins(), ins);

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

  zero_bias = std::make_shared<std::vector<float>>(oc, 0.0f);

  // in int8 case, bias will be add after mkldnn conv
  // reason is int8 case, bias format is 32bit
  bool do_bias = with_bias;
  if (!do_bias) {
    bias_data = zero_bias;
  }

  output_data = this->resTensor;
#define CONV3D_USE_MKLDNN (0)

  // set mkldnn
  if (CONV3D_USE_MKLDNN) {
    using namespace mkldnn;
    this->mkl_eng = mkldnn::engine(mkldnn::engine::kind::cpu, 0);
    this->mkl_stream = mkldnn::stream(mkl_eng);
    using tag = memory::format_tag;
    using dt = memory::data_type;

    engine eng(engine::kind::cpu, 0);
    stream s(eng);

    const memory::dim batch = n;
    memory::dims src_tz = { batch, ic, id, ih, iw };
    // setting comes from https://github.com/intel/caffe/blob/master/src/caffe/layers/mkldnn_convolution_layer.cpp:29
    memory::dims weights_tz = (g != 1) ? memory::dims{g, oc/g, ic/g, kd, kh, kw}
    : memory::dims{oc, ic, kd, kh, kw};
    memory::dims bias_tz = { oc };
    memory::dims dst_tz = { batch, oc, od, oh, ow };
    memory::dims strides = { sd, sh, sw };

    memory::dims padding_l = { pd0, pt, pl };
    memory::dims padding_r = { pd1, pb, pr };
    memory::dims dilation = { dd-1, dh-1, dw-1 };

    // memory
    auto user_src_memory = memory(
        { { src_tz }, dt::f32, tag::ncdhw }, eng, input_data->data());
    auto user_weights_memory = (g != 1)
      ? memory({ { weights_tz }, dt::f32, tag::goidhw }, eng, filter_data->data())
      : memory({ { weights_tz }, dt::f32, tag::oidhw }, eng, filter_data->data());
    auto user_bias_memory = memory(
        { { bias_tz }, dt::f32, tag::x }, eng, bias_data->data());
    auto user_dst_memory = memory(
        { { dst_tz }, dt::f32, tag::ncdhw }, eng, output_data->data());

    // md
    auto src_md     = memory::desc({ src_tz }, dt::f32, tag::any);
    auto weights_md = memory::desc({ weights_tz }, dt::f32, tag::any);
    auto bias_md    = memory::desc({ bias_tz }, dt::f32, tag::any);
    auto dst_md     = memory::desc({ dst_tz }, dt::f32, tag::any);

    // conv desc
    auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
        algorithm::convolution_direct, src_md, weights_md, bias_md, dst_md,
        strides, dilation, padding_l, padding_r);
    auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, eng);

    // do reorder if needed
    auto src_memory = user_src_memory;
    if (conv_prim_desc.src_desc() != user_src_memory.get_desc()) {
      src_memory = memory(conv_prim_desc.src_desc(), eng);
      mkl_net.push_back(reorder(user_src_memory, src_memory));
      mkl_net_args.push_back({ { MKLDNN_ARG_FROM, user_src_memory },
          { MKLDNN_ARG_TO, src_memory } });
    }
    auto weights_memory = user_weights_memory;
    if (conv_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
      weights_memory = memory(conv_prim_desc.weights_desc(), eng);
      reorder(user_weights_memory, weights_memory)
        .execute(s, user_weights_memory, weights_memory);
    }
    auto bias_memory = user_bias_memory;

    auto dst_memory = memory(conv_prim_desc.dst_desc(), eng);

    mkl_net.push_back(convolution_forward(conv_prim_desc));
    mkl_net_args.push_back({ { MKLDNN_ARG_SRC, src_memory },
        { MKLDNN_ARG_WEIGHTS, weights_memory },
        { MKLDNN_ARG_BIAS, bias_memory },
        { MKLDNN_ARG_DST, dst_memory } });

    // reorder or copy the output
    if (dst_memory != user_dst_memory) {
      mkl_net.push_back(reorder(dst_memory, user_dst_memory));
      mkl_net_args.push_back({ { MKLDNN_ARG_FROM, dst_memory },
          { MKLDNN_ARG_TO, user_dst_memory } });
    }

    // run
    assert(mkl_net.size() == mkl_net_args.size() && "something is missing");
  }
  else {
    // leverage cpu
  }
}

void Conv3DOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " Conv op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> Conv3DOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void Conv3DOpKernel::fp32_invoke() {
  if (CONV3D_USE_MKLDNN) {
    try{

      for (size_t i = 0; i < mkl_net.size(); ++i) {
        mkl_net.at(i).execute(mkl_stream, mkl_net_args.at(i));
      }
      mkl_stream.wait();
    } catch (mkldnn::error &e) {
      llvm::errs() << "status: " << e.status << "\n";
      llvm::errs() << "message: " << e.message << "\n";
      llvm_unreachable("wrong config of mkldnn");
    }
  }
  else {
    float* _bias = with_bias ? bias_data->data() : NULL;
    _conv3d_float_ref(input_data->data(), filter_data->data(), _bias, output_data->data(),
        n, ic, id, ih, iw,
        oc, od, oh, ow,
        kd, kh, kw,
        sd, sh, sw,
        dd, dh, dw,
        pd0, pt, pb, pd1, pl, pr);
  }

  if (do_relu) {
    relu(output_data->data(), output_data->data(), output_data->size());
  }
};

void Conv3DOpKernel::i8_invoke() {
  // mkldnn not support non zero padding
  // we handle it.
  if (pad_value != 0 && is_asymmetric) {
    llvm_unreachable("not support asymmetric");
    //once_mkldnn_conv(input_data->data(), filter_data->data(),
    //                 use_multiplier ? nullptr : bias_data->data(),
    //                 output_data->data(), n, ic, ih, iw, oc, oh, ow, kh, kw, sh,
    //                 sw, dh, dw, pt, pb, pl, pr, g, pad_value);

    if (is_perchannel) {
      if (use_multiplier) {
        quantizeActivationInt8PerChannelMultiplierAndRShift(
            output_data->data(), output_data->data(), bias_data->data(), false,
            n, oc, oh * ow, rshift.data(), multiplier.data());
      } else {
        quantizeActivationInt8PerChannelRShift(output_data->data(),
                                               output_data->data(), n, oc,
                                               oh * ow, rshift.data());
      }
    } else {
      quantizeActivationInt8PerLayerRshift(output_data->data(),
                                           output_data->data(),
                                           output_data->size(), rshift.at(0));
    }
    return;
  }

  if (CONV3D_USE_MKLDNN) {
    for (size_t i = 0; i < mkl_net.size(); ++i) {
      mkl_net.at(i).execute(mkl_stream, mkl_net_args.at(i));
    }
    mkl_stream.wait();
  }
  else {
    float* _bias = with_bias ? bias_data->data() : NULL;
    _conv3d_float_ref(input_data->data(), filter_data->data(), _bias, output_data->data(),
        n, ic, id, ih, iw,
        oc, od, oh, ow,
        kd, kh, kw,
        sd, sh, sw,
        dd, dh, dw,
        pd0, pt, pb, pd1, pl, pr);
  }

  if (is_perchannel) {
    if (use_multiplier) {
      quantizeActivationInt8PerChannelMultiplierAndRShift(
          output_data->data(), output_data->data(), bias_data->data(), do_relu,
          n, oc, oh * ow, rshift.data(), multiplier.data());
    } else {
      if (do_relu && !is_asymmetric) {
        relu(output_data->data(), output_data->data(), output_data->size());
      }
      quantizeActivationInt8PerChannelRShift(output_data->data(),
                                             output_data->data(), n, oc,
                                             oh * ow, rshift.data());
    }
  } else {
    if (do_relu && !is_asymmetric) {
      relu(output_data->data(), output_data->data(), output_data->size());
    }
    quantizeActivationInt8PerLayerRshift(output_data->data(),
                                         output_data->data(),
                                         output_data->size(), rshift.at(0));
  }
};

void Conv3DOpKernel::invoke() {
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

void Conv3DOpKernel::dump() {
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

}