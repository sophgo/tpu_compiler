#ifndef MLIR_DIALECT_TPU_NATIVE_CPU_IMPLEMENTATION_H_
#define MLIR_DIALECT_TPU_NATIVE_CPU_IMPLEMENTATION_H_

#include "mkldnn.hpp"
#include <unordered_map>

using tag = mkldnn::memory::format_tag;
using dt = mkldnn::memory::data_type;
using cmdargs = std::unordered_map<int, mkldnn::memory>;

class MKLConv {
public:
  MKLConv() {
    eng = mkldnn::engine(mkldnn::engine::kind::cpu, 0);
    stream = mkldnn::stream(eng);
  }

  void setup(float *weight, float *bias, int n, int ic, int ih, int iw, int oc,
             int oh, int ow, int kh, int kw, int sh, int sw, int dh, int dw,
             int pt, int pb, int pl, int pr, int g) {
    src_shape = {n, ic, ih, iw};
    dst_shape = {n, oc, oh, ow};
    mkldnn::memory::dims filter_shape =
        (g != 1) ? mkldnn::memory::dims{g, oc / g, ic / g, kh, kw}
                 : mkldnn::memory::dims{oc, ic, kh, kw};
    mkldnn::memory::dims bias_shape = {oc};
    mkldnn::memory::dims strides = {sh, sw};

    mkldnn::memory::dims padding_l = {pt, pl};
    mkldnn::memory::dims padding_r = {pb, pr};
    // mkldnn dialtion is different with caffe
    mkldnn::memory::dims dilation = {dh - 1, dw - 1};

    auto src_md = mkldnn::memory::desc({src_shape}, dt::f32, tag::any);
    auto filter_md = mkldnn::memory::desc({filter_shape}, dt::f32, tag::any);
    auto bias_md = mkldnn::memory::desc({bias_shape}, dt::f32, tag::any);
    auto dst_md = mkldnn::memory::desc({dst_shape}, dt::f32, tag::any);

    auto conv_desc = mkldnn::convolution_forward::desc(
        mkldnn::prop_kind::forward_inference,
        mkldnn::algorithm::convolution_direct, src_md, filter_md, bias_md,
        dst_md, strides, dilation, padding_l, padding_r);

    conv_prim_desc =
        mkldnn::convolution_forward::primitive_desc(conv_desc, eng);

    // set mkldnn memory
    auto filter_tag = (g != 1) ? tag::goihw : tag::oihw;
    auto filter_memory =
        mkldnn::memory({{filter_shape}, dt::f32, filter_tag}, eng, weight);
    prim_filter_memory = filter_memory;
    if (conv_prim_desc.weights_desc() != filter_memory.get_desc()) {
      prim_filter_memory = mkldnn::memory(conv_prim_desc.weights_desc(), eng);
      mkldnn::reorder(filter_memory, prim_filter_memory)
          .execute(stream, filter_memory, prim_filter_memory);
    }
    prim_bias_memory =
        mkldnn::memory({{bias_shape}, dt::f32, tag::x}, eng, bias);
  }

  void run(float *input, float *output) {
    std::vector<mkldnn::primitive> net;
    std::vector<std::unordered_map<int, mkldnn::memory>> net_args;
    // do reorder if needed
    auto src_memory =
        mkldnn::memory({{src_shape}, dt::f32, tag::nchw}, eng, input);
    auto prim_src_memory = src_memory;
    if (conv_prim_desc.src_desc() != src_memory.get_desc()) {
      prim_src_memory = mkldnn::memory(conv_prim_desc.src_desc(), eng);
      net.push_back(mkldnn::reorder(src_memory, prim_src_memory));
      net_args.push_back(
          {{MKLDNN_ARG_FROM, src_memory}, {MKLDNN_ARG_TO, prim_src_memory}});
    }

    auto prim_dst_memory = mkldnn::memory(conv_prim_desc.dst_desc(), eng);
    net.push_back(mkldnn::convolution_forward(conv_prim_desc));
    net_args.push_back({{MKLDNN_ARG_SRC, prim_src_memory},
                        {MKLDNN_ARG_WEIGHTS, prim_filter_memory},
                        {MKLDNN_ARG_BIAS, prim_bias_memory},
                        {MKLDNN_ARG_DST, prim_dst_memory}});
    // reorder or copy the output
    auto dst_memory =
        mkldnn::memory({{dst_shape}, dt::f32, tag::nchw}, eng, output);
    if (prim_dst_memory != dst_memory) {
      net.push_back(mkldnn::reorder(prim_dst_memory, dst_memory));
      net_args.push_back(
          {{MKLDNN_ARG_FROM, prim_dst_memory}, {MKLDNN_ARG_TO, dst_memory}});
    }
    for (size_t i = 0; i < net.size(); ++i)
      net.at(i).execute(stream, net_args.at(i));
    stream.wait();
  }

private:
  mkldnn::engine eng;
  mkldnn::stream stream;
  mkldnn::convolution_forward::primitive_desc conv_prim_desc;
  mkldnn::memory prim_filter_memory;
  mkldnn::memory prim_bias_memory;
  mkldnn::memory::dims src_shape;
  mkldnn::memory::dims dst_shape;
};

class MKLPooling {
public:
  MKLPooling() {
    eng = mkldnn::engine(mkldnn::engine::kind::cpu, 0);
    stream = mkldnn::stream(eng);
  }

  void setup(int n, int c, int ih, int iw, int oh, int ow, int kh, int kw,
             int sh, int sw, int pt, int pb, int pl, int pr, bool is_avg,
             bool count_include_pad, int pad_value = 0) {

    src_shape = {n, c, ih, iw};
    dst_shape = {n, c, oh, ow};
    mkldnn::memory::dims strides = {sh, sw};
    mkldnn::memory::dims kernel = {kh, kw};
    mkldnn::memory::dims padding_tl = {pt, pl};
    mkldnn::memory::dims padding_br = {pb, pr};
    auto src_md = mkldnn::memory::desc({src_shape}, dt::f32, tag::nchw);
    auto dst_md = mkldnn::memory::desc({dst_shape}, dt::f32, tag::nchw);
    auto pool_avg_algo = count_include_pad
                             ? mkldnn::algorithm::pooling_avg_include_padding
                             : mkldnn::algorithm::pooling_avg_exclude_padding;
    // pool desc
    auto pool_desc = mkldnn::pooling_forward::desc(
        mkldnn::prop_kind::forward_inference,
        is_avg ? pool_avg_algo : mkldnn::algorithm::pooling_max, src_md, dst_md,
        strides, kernel, padding_tl, padding_br);

    prim_desc = mkldnn::pooling_forward::primitive_desc(pool_desc, eng);
  }

  void run(float *input, float *output) {
    std::vector<mkldnn::primitive> net;
    std::vector<std::unordered_map<int, mkldnn::memory>> net_args;
    // do reorder if needed
    mkldnn::memory src_memory =
        mkldnn::memory({{src_shape}, dt::f32, tag::nchw}, eng, input);

    auto prim_src_memory = src_memory;
    if (prim_desc.src_desc() != src_memory.get_desc()) {
      prim_src_memory = mkldnn::memory(prim_desc.src_desc(), eng);
      net.push_back(mkldnn::reorder(src_memory, prim_src_memory));
      net_args.push_back(
          {{MKLDNN_ARG_FROM, src_memory}, {MKLDNN_ARG_TO, prim_src_memory}});
    }

    auto prim_dst_memory = mkldnn::memory(prim_desc.dst_desc(), eng);
    net.push_back(mkldnn::pooling_forward(prim_desc));
    net_args.push_back(
        {{MKLDNN_ARG_SRC, prim_src_memory}, {MKLDNN_ARG_DST, prim_dst_memory}});

    // reorder or copy the output
    mkldnn::memory dst_memory =
        mkldnn::memory({{dst_shape}, dt::f32, tag::nchw}, eng, output);
    if (prim_dst_memory != dst_memory) {
      net.push_back(mkldnn::reorder(prim_dst_memory, dst_memory));
      net_args.push_back(
          {{MKLDNN_ARG_FROM, prim_dst_memory}, {MKLDNN_ARG_TO, dst_memory}});
    }
    for (size_t i = 0; i < net.size(); ++i)
      net.at(i).execute(stream, net_args.at(i));
    stream.wait();
  }

private:
  mkldnn::engine eng;
  mkldnn::stream stream;
  mkldnn::pooling_forward::primitive_desc prim_desc;
  mkldnn::memory::dims src_shape;
  mkldnn::memory::dims dst_shape;
};

class MKLInt8AvgPooling : public MKLConv {
public:
  MKLInt8AvgPooling() : MKLConv() {}

  void setup(int n, int c, int ih, int iw, int oh, int ow, int kh, int kw,
             int sh, int sw, int pt, int pb, int pl, int pr) {
    size_t filter_size = c * kh * kw;
    filter_data = std::make_unique<std::vector<float>>(filter_size, 1.0f);
    zero_bias = std::make_unique<std::vector<float>>(c, 0.0f);
    MKLConv::setup(filter_data->data(), zero_bias->data(), n, c, ih, iw, c, oh,
                   ow, kh, kw, sh, sw, 1, 1, pt, pb, pl, pr, c);
  }

private:
  std::unique_ptr<std::vector<float>> filter_data;
  std::unique_ptr<std::vector<float>> zero_bias;
};

class MKLDeconv {
public:
  MKLDeconv() {
    eng = mkldnn::engine(mkldnn::engine::kind::cpu, 0);
    stream = mkldnn::stream(eng);
  }

  void setup(float *weight, float *bias, int n, int ic, int ih, int iw, int oc,
             int oh, int ow, int kh, int kw, int sh, int sw, int dh, int dw,
             int pt, int pb, int pl, int pr, int g) {
    src_shape = {n, ic, ih, iw};
    dst_shape = {n, oc, oh, ow};
    mkldnn::memory::dims filter_shape =
        (g != 1) ? mkldnn::memory::dims{g, oc / g, ic / g, kh, kw}
                 : mkldnn::memory::dims{oc, ic, kh, kw};
    mkldnn::memory::dims bias_shape = {oc};
    mkldnn::memory::dims strides = {sh, sw};

    mkldnn::memory::dims padding_l = {pt, pl};
    mkldnn::memory::dims padding_r = {pb, pr};
    // mkldnn dialtion is different with caffe
    mkldnn::memory::dims dilation = {dh - 1, dw - 1};

    auto src_md = mkldnn::memory::desc({src_shape}, dt::f32, tag::any);
    auto filter_md = mkldnn::memory::desc({filter_shape}, dt::f32, tag::any);
    auto bias_md = mkldnn::memory::desc({bias_shape}, dt::f32, tag::any);
    auto dst_md = mkldnn::memory::desc({dst_shape}, dt::f32, tag::any);

    auto deconv_desc = mkldnn::deconvolution_forward::desc(
        mkldnn::prop_kind::forward_inference,
        mkldnn::algorithm::deconvolution_direct, src_md, filter_md, bias_md,
        dst_md, strides, dilation, padding_l, padding_r);
    deconv_prim_desc =
        mkldnn::deconvolution_forward::primitive_desc(deconv_desc, eng);

    // set mkldnn memory
    auto filter_memory =
        (g != 1)
            ? mkldnn::memory({{filter_shape}, dt::f32, tag::goihw}, eng, weight)
            : mkldnn::memory({{filter_shape}, dt::f32, tag::oihw}, eng, weight);
    prim_filter_memory = filter_memory;
    if (deconv_prim_desc.weights_desc() != filter_memory.get_desc()) {
      prim_filter_memory = mkldnn::memory(deconv_prim_desc.weights_desc(), eng);
      mkldnn::reorder(filter_memory, prim_filter_memory)
          .execute(stream, filter_memory, prim_filter_memory);
    }
    prim_bias_memory =
        mkldnn::memory({{bias_shape}, dt::f32, tag::x}, eng, bias);
  }

  void run(float *input, float *output) {
    // do reorder if needed
    std::vector<mkldnn::primitive> net;
    std::vector<std::unordered_map<int, mkldnn::memory>> net_args;
    auto src_memory =
        mkldnn::memory({{src_shape}, dt::f32, tag::nchw}, eng, input);

    auto prim_src_memory = src_memory;
    if (deconv_prim_desc.src_desc() != src_memory.get_desc()) {
      prim_src_memory = mkldnn::memory(deconv_prim_desc.src_desc(), eng);
      net.push_back(mkldnn::reorder(src_memory, prim_src_memory));
      net_args.push_back(
          {{MKLDNN_ARG_FROM, src_memory}, {MKLDNN_ARG_TO, prim_src_memory}});
    }

    auto prim_dst_memory = mkldnn::memory(deconv_prim_desc.dst_desc(), eng);
    net.push_back(mkldnn::deconvolution_forward(deconv_prim_desc));
    net_args.push_back({{MKLDNN_ARG_SRC, prim_src_memory},
                        {MKLDNN_ARG_WEIGHTS, prim_filter_memory},
                        {MKLDNN_ARG_BIAS, prim_bias_memory},
                        {MKLDNN_ARG_DST, prim_dst_memory}});

    // reorder or copy the output
    auto dst_memory =
        mkldnn::memory({{dst_shape}, dt::f32, tag::nchw}, eng, output);
    if (prim_dst_memory != dst_memory) {
      net.push_back(mkldnn::reorder(prim_dst_memory, dst_memory));
      net_args.push_back(
          {{MKLDNN_ARG_FROM, prim_dst_memory}, {MKLDNN_ARG_TO, dst_memory}});
    }
    for (size_t i = 0; i < net.size(); ++i)
      net.at(i).execute(stream, net_args.at(i));
    stream.wait();
  }

private:
  mkldnn::engine eng;
  mkldnn::stream stream;
  mkldnn::deconvolution_forward::primitive_desc deconv_prim_desc;
  mkldnn::memory prim_filter_memory;
  mkldnn::memory prim_bias_memory;
  mkldnn::memory::dims src_shape;
  mkldnn::memory::dims dst_shape;
};

//
// mkldnn functions
//
int mkldnn_conv(float *input, float *weight, float *bias, float *output, int n,
                int ic, int ih, int iw, int oc, int oh, int ow, int kh, int kw,
                int sh, int sw, int dh, int dw, int pt, int pb, int pl, int pr,
                int g, int pad_value);

int mkldnn_ip(float *input, float *weight, float *bias, float *output, int m,
              int k, int n, bool transpose);

//
// native cpu functions
//

int calc_dilute_hw(int h, int ins_h, int ins_h_l, int pad_h_b, int pad_h_t);
void my_dilateActivation(float *input, float *output, int pad_h_t, int pad_h_b,
                         int ins_h, int ins_h_l, int pad_w_l, int pad_w_r,
                         int ins_w, int ins_w_l, int n, int c, int h, int w,
                         int fill_constant = 0);

void my_interp(const int channels, const float *data1, const int x1,
               const int y1, const int height1, const int width1,
               const int Height1, const int Width1, float *data2, const int x2,
               const int y2, const int height2, const int width2,
               const int Height2, const int Width2);

int my_scale(float *input, float *scale, float *bias, float *output, int n,
             int c, int h, int w);

int my_pixelshuffle(float *input, float *output, int in, int ic, int ih, int iw,
                    int on, int oc, int oh, int ow, int upscale_factor,
                    bool dcr_mode = false);

int my_upsample(float *input, float *output, int n, int c, int ih, int iw,
                int scale_h, int scale_w);

int my_permute(float *input, float *output, int in, int ic, int ih, int iw,
               int order0, int order1, int order2, int order3);

float my_mish_activate(float x_val);

int my_pad_constant(float *input, float *output,
                    std::vector<int64_t> &input_shape, std::vector<int> &pads,
                    float const_val);

void conv3d_float_ref(float *input, float *weight, float *bias, float *output,
                      int n, int input_c, int input_d, int input_h, int input_w,
                      int output_c, int output_d, int output_h, int output_w,
                      int kernel_d, int kernel_h, int kernel_w, int stride_d,
                      int stride_h, int stride_w, int dilation_d,
                      int dilation_h, int dilation_w, int pad_d0, int pad_top,
                      int pad_bottom, int pad_d1, int pad_left, int pad_right);

void pool3d_float_ref(float *input, float *output, int input_n, int input_c,
                      int input_d, int input_h, int input_w, int output_d,
                      int output_h, int output_w, int kernel_d, int kernel_h,
                      int kernel_w, int stride_d, int stride_h, int stride_w,
                      int pad_d0, int pad_d1, int pad_top, int pad_bot,
                      int pad_left, int pad_right);

#endif // MLIR_DIALECT_TPU_NATIVE_CPU_IMPLEMENTATION_H_
