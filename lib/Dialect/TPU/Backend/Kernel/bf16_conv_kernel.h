/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: bf16_conv_kernel.h
 * Description:
 */

#ifndef BF16_CONV_KERNEL_H_
#define BF16_CONV_KERNEL_H_

#include "CviBackendContext.h"
// FIXME: test strategy order, current order is split oc -> reuse weight -> split ic
typedef enum {
  POLICY_SPLIT_OC = 1 << 1,
  POLICY_NO_REUSE_WEIGHT = 1 << 2,
  POLICY_NO_SPLIT = 1 << 3,
  POLICY_SPLIT_IC = 1 << 4,
} ConvBF16_POLICY; // policy means strategy for slicing

typedef struct {
  gaddr_t ga_ifmap;
  gaddr_t ga_ofmap;
  gaddr_t ga_weight;
  gaddr_t ga_bias;
  gaddr_t ga_bn_mean;
  gaddr_t ga_bn_variance;
  gaddr_t ga_scale;
  gaddr_t ga_scale_bias;
  int input_n;
  int input_c;
  int input_h;
  int input_w;
  int groups;
  int output_c;
  uint16_t kh;
  uint16_t kw;
  uint16_t dilation_h;
  uint16_t dilation_w;
  uint8_t pad_top;
  uint8_t pad_bottom;
  uint8_t pad_left;
  uint8_t pad_right;
  uint8_t stride_h;
  uint8_t stride_w;
  bool do_bias;
  bool do_bn;
  bool do_scale;
  bool do_scale_bias;
  bool do_activation;
  float bn_scale;
  float bn_eps;
  int activation_method;
  float *activation_arg;
  gaddr_t activation_ga_slope;
  uint32_t layer_id;
  ConvBF16_POLICY policy;
} ConvBF16_ARGS;


typedef struct {
  int n;
  int oc;
  int ic;
  int h;
  int w;
  int ic_step;
  int oc_step;
  int oh_step;
  int ow_step;
  int ih_step;
  int iw_step;
} SLICES;

class BM1880v2ConvBF16 {
 public:
  BM1880v2ConvBF16(ConvBF16_ARGS &args) {
    slices.n = 1;
    slices.oc = 1;
    slices.ic = 1;
    slices.h = 1;
    slices.oc_step = args.output_c;
    slices.oh_step = 1;
    slices.ih_step = args.input_h;
    ga_ifmap = args.ga_ifmap;
    ga_ofmap = args.ga_ofmap;
    ga_weight = args.ga_weight;
    ga_bias = args.ga_bias;
    ga_bn_mean = args.ga_bn_mean;
    ga_bn_variance = args.ga_bn_variance;
    ga_scale = args.ga_scale;
    ga_scale_bias = args.ga_scale_bias;
    input_n = args.input_n;
    input_c = args.input_c;
    input_h = args.input_h;
    input_w = args.input_w;
    groups = args.groups;
    output_c = args.output_c;
    kh = args.kh;
    kw = args.kw;
    dilation_h = args.dilation_h;
    dilation_w = args.dilation_w;
    pad_top = args.pad_top;
    pad_bottom = args.pad_bottom;
    pad_left = args.pad_left;
    pad_right = args.pad_right;
    stride_h = args.stride_h;
    stride_w = args.stride_w;
    do_bias = args.do_bias;
    do_bn = args.do_bn;
    do_scale = args.do_scale;
    do_scale_bias = args.do_scale_bias;
    do_activation = args.do_activation;
    bn_scale = args.bn_scale;
    bn_eps = args.bn_eps;
    activation_method = args.activation_method;
    activation_arg = args.activation_arg;
    activation_ga_slope = args.activation_ga_slope;
    layer_id = args.layer_id;
    policy = POLICY_NO_SPLIT;
  }

  void do_conv(const CviBackendContext &ctx);
  int split(const CviBackendContext &ctx);
  int _split(const CviBackendContext &ctx, int duplicate_weights);
  int split_ic(const CviBackendContext &ctx);
  int _split_oc(const CviBackendContext &ctx, int ic_step);
  int _split_ic(const CviBackendContext &ctx, int ic_step, int oc_step);
  void ConvReuseActivation(const CviBackendContext &ctx);
  void ConvReuseWeight(const CviBackendContext &ctx);
  void ConvPs32(const CviBackendContext &ctx);
  void DepthwiseConv(const CviBackendContext &ctx);
  int conv_ps(
    const CviBackendContext &ctx,
    cvk_tiu_pt_convolution_param_t* conv_param,
    uint32_t ic_step, uint32_t oc_step, uint32_t n_step,
    cvk_tg_shape_t weight_shape,
    gaddr_t input_gaddr, gaddr_t weight_gaddr,
    cvk_fmt_t fmt);

  void tl_copy(const CviBackendContext &ctx,
      cvk_tl_t* dst,
      const cvk_tl_t* src,
      int re_n, int re_c, int re_h, int re_w, uint8_t eu_align);
  void tl_reshape(const CviBackendContext &ctx,
      cvk_tl_t* dst,
      int re_n, int re_c, int re_h, int re_w, uint8_t eu_align);
  cvk_tl_t* conv_ifmap_tensor(
      const CviBackendContext &ctx,
      int input_n, int input_c, int input_h, int input_w,
      uint32_t ic_step, cvk_fmt_t fmt);
  cvk_tl_t* conv_weight_tensor(
      const CviBackendContext &ctx,
      int kh, int kw, uint32_t ic_step, uint32_t oc_step,
      int output_c, int input_c,
      cvk_fmt_t fmt);
  cvk_tl_t* conv_ofmap_tensor(
      const CviBackendContext &ctx,
      int input_n, int oh, int ow,
      int output_c,
      uint32_t oc_step, cvk_fmt_t fmt);
  cvk_tl_t* conv_bias_tensor(
      const CviBackendContext &ctx,
      int output_c, cvk_fmt_t fmt);

  bool is_reuse_weight() {
    return !(policy & POLICY_NO_REUSE_WEIGHT);
  }
  bool is_split_oc() {
    return (policy & POLICY_SPLIT_OC);
  }
  bool is_split_ic() {
    return (policy & POLICY_SPLIT_IC);
  }

  ~BM1880v2ConvBF16() {}

 protected:
  SLICES slices;
  gaddr_t ga_ifmap;
  gaddr_t ga_ofmap;
  gaddr_t ga_weight;
  gaddr_t ga_bias;
  gaddr_t ga_bn_mean;
  gaddr_t ga_bn_variance;
  gaddr_t ga_scale;
  gaddr_t ga_scale_bias;
  int input_n;
  int input_c;
  int input_h;
  int input_w;
  int groups;
  int output_c;
  uint16_t kh;
  uint16_t kw;
  uint16_t dilation_h;
  uint16_t dilation_w;
  uint8_t pad_top;
  uint8_t pad_bottom;
  uint8_t pad_left;
  uint8_t pad_right;
  uint8_t stride_h;
  uint8_t stride_w;
  bool result_add;
  bool do_bias;
  bool do_bn;
  bool do_scale;
  bool do_scale_bias;
  bool do_activation;
  float bn_scale;
  float bn_eps;
  int activation_method;
  float *activation_arg;
  gaddr_t activation_ga_slope;
  uint32_t layer_id;
  int policy;
};

#endif // BF16_CONV_KERNEL_H_
