/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
//#include <targets/plat-bm188x/bmkernel/bmkernel_api.h>
//#include <backend/TensorInst.hpp>
//#include <targets/Target.hpp>
//#include <utils/common.hpp>
#include <bmkernel/bm_kernel.h>
#include <bmkernel/bm_kernel_legacy.h>
#include "BM1880v2BackendContext.h"

#define DEBUG_CONV

//namespace bmnet {

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
  u16 kh;
  u16 kw;
  u16 dilation_h;
  u16 dilation_w;
  u8 pad_top;
  u8 pad_bottom;
  u8 pad_left;
  u8 pad_right;
  u8 stride_h;
  u8 stride_w;
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
  bool activation_channel_shared;
  int activation_gt_scale;
  int activation_gt_rshift;
  int activation_le_scale;  // slope; TODO
  int activation_le_rshift;
  int right_shift_width;
  int bn_right_shift_width;
  int scale_right_shift_width;
  u32 layer_id;
} ConvFixed_ARGS;

typedef struct {
  int n;
  int oc;
  int ic;
  int h;
  int w;
} SLICES;

class BM1880v2ConvFixed {
 public:
  BM1880v2ConvFixed(ConvFixed_ARGS &args) {
    slices.n = 1;
    slices.oc = 1;
    slices.ic = 1;
    slices.h = 1;
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
    result_add = args.result_add;
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
    activation_channel_shared = args.activation_channel_shared;
    activation_gt_scale = args.activation_gt_scale;
    activation_gt_rshift = args.activation_gt_rshift;
    activation_le_scale = args.activation_le_scale;
    activation_le_rshift = args.activation_le_rshift;
    right_shift_width = args.right_shift_width;
    bn_right_shift_width = args.bn_right_shift_width;
    scale_right_shift_width = args.scale_right_shift_width;
    layer_id = args.layer_id;
  }

  virtual void do_conv(const BM1880v2BackendContext &ctx) = 0;
  virtual int split(const BM1880v2BackendContext &ctx) = 0;
  virtual ~BM1880v2ConvFixed() {}

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
  u16 kh;
  u16 kw;
  u16 dilation_h;
  u16 dilation_w;
  u8 pad_top;
  u8 pad_bottom;
  u8 pad_left;
  u8 pad_right;
  u8 stride_h;
  u8 stride_w;
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
  bool activation_channel_shared;
  int activation_gt_scale;
  int activation_gt_rshift;
  int activation_le_scale;
  int activation_le_rshift;
  int right_shift_width;
  int bn_right_shift_width;
  int scale_right_shift_width;
  u32 layer_id;
};

class BM1880v2ConvFixedParallel : public BM1880v2ConvFixed {
 public:
  BM1880v2ConvFixedParallel(ConvFixed_ARGS &args) : BM1880v2ConvFixed(args) {}
  void do_conv(const BM1880v2BackendContext &ctx);
  int split(const BM1880v2BackendContext &ctx);
  ~BM1880v2ConvFixedParallel() {}
};

class BM1880v2ConvFixedParallelv2 : public BM1880v2ConvFixed {
 public:
  BM1880v2ConvFixedParallelv2(ConvFixed_ARGS &args) : BM1880v2ConvFixed(args) {}
  void do_conv(const BM1880v2BackendContext &ctx);
  int split(const BM1880v2BackendContext &ctx);
  ~BM1880v2ConvFixedParallelv2() {}
};

class BM1880v2ConvFixedParallelv2_qdm : public BM1880v2ConvFixed {
 public:
  BM1880v2ConvFixedParallelv2_qdm(ConvFixed_ARGS &args) : BM1880v2ConvFixed(args) {}
  void do_conv(const BM1880v2BackendContext &ctx);
  int split(const BM1880v2BackendContext &ctx);
  ~BM1880v2ConvFixedParallelv2_qdm() {}
};

class BM1880v2ConvFixedSerial : public BM1880v2ConvFixed {
 public:
  BM1880v2ConvFixedSerial(ConvFixed_ARGS &args) : BM1880v2ConvFixed(args) {}
  void do_conv(const BM1880v2BackendContext &ctx);
  int split(const BM1880v2BackendContext &ctx);
  ~BM1880v2ConvFixedSerial() {}
};

//}  // namespace bmnet
