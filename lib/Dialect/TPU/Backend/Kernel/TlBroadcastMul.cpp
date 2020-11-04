/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: tl_broadcast_mul.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include "backend/backend_tl_api.h"

#define DEBUG_TYPE "tl_broadcast_mul"
#define MAX_W (1 << 11)

// with no bias
void cvi_backend_tl_broadcast_mul(
      const CviBackendContext &ctx, uint32_t layer_id,
      laddr_t input_laddr, laddr_t scale_laddr,
      laddr_t bias_laddr, laddr_t output_laddr, int input_n,
      int input_c, int input_h, int input_w, int scale_dim,
      int inner_dim, bool is_scale_const, int const_scale,
      int right_shift_width,
      int do_activation,
      int activation_method,
      float activation_arg[],
      const int *i8_multiplier, // INT8_PER_LAYER
      bool do_bias) {
  assert(!do_bias && "unsupport bias");
  // output
  cvk_tl_shape_t tl_output_shape = ctx.tl_shape_t4(input_n, input_c, input_h, input_w);
  cvk_tl_t tl_output;
  tl_output.start_address = output_laddr;
  tl_output.fmt = CVK_FMT_I8;
  tl_output.shape = tl_output_shape;
  tl_output.stride = ctx.tl_default_stride(tl_output_shape, CVK_FMT_I8, 1);


#define RELU (0)
  bool fused_relu = (do_activation && activation_method == RELU && (activation_arg[0] == 0.0f));
  LLVM_DEBUG(llvm::errs() << "fused_relu is " << fused_relu;);

  if (is_scale_const) {
    LLVM_DEBUG(llvm::errs() << llvm::format(
                               "quantization_fixed_forward_bmkernel:\n"
                               " layer_id, input_laddr 0x%lx, output_laddr 0x%lx\n"
                               "  in(%d, %d, %d, %d), quantization_num %d\n",
                               layer_id, input_laddr, output_laddr, input_n, input_c,
                               input_h, input_w));
    LLVM_DEBUG(llvm::errs() << "quantization i8_multiplier=" << const_scale
                          << ",right_shift_width=" << right_shift_width << "\n");

    int sec_n = input_n;
    int count = ALIGN(sec_n * input_c * input_h * input_w, MAX_W);
    cvk_ml_shape_t shape_ = ctx.ml_shape_t1(count, CVK_FMT_I8);
    cvk_tl_t tl_input;
    tl_input.start_address = input_laddr;
    tl_input.fmt = CVK_FMT_I8;
    tl_input.shape = {shape_.n, shape_.c, 1, shape_.w};
    tl_input.stride = ctx.tl_default_stride(tl_input.shape, CVK_FMT_I8,
                                            /*eu_aligned=*/1);

    // FIXME: need to verify INT8_PER_LAYER value
    cvk_tiu_mul_param_t p = {0};
    p.res_high = nullptr;
    p.res_low = &tl_output;
    p.a = &tl_input;
    p.b_is_const = 1;
    p.b_const.val = const_scale * i8_multiplier[0];
    p.b_const.is_signed = false;
    p.rshift_bits = right_shift_width;
    p.layer_id = layer_id;
    p.relu_enable = fused_relu;
    ctx.tiu_mul(&p);
    return;
  }
  LLVM_DEBUG(llvm::errs() << llvm::format(
      "input_laddr 0x%lx scale_laddr 0x%lx, "
      "bias_laddr:0x%lx, output_laddr:0x%lx, "
      "input_n %d input_c %d input_h %d input_w %d "
      "scale_dim %d inner_dim %d \n",
      input_laddr, scale_laddr, bias_laddr, output_laddr, input_n, input_c,
      input_h, input_w, scale_dim, inner_dim););

  assert(input_n * input_c * input_h * input_w == scale_dim * inner_dim);

  // input
  cvk_tl_shape_t tl_input_shape = ctx.tl_shape_t4(input_n, input_c, input_h, input_w);
  cvk_tl_t tl_input;
  tl_input.start_address = input_laddr;
  tl_input.fmt = CVK_FMT_I8;
  tl_input.shape = tl_input_shape;
  tl_input.stride = ctx.tl_default_stride(tl_input_shape, CVK_FMT_I8, 1);

  // scale
  cvk_tl_shape_t tl_scale_shape;
  cvk_tl_t tl_scale;
  tl_scale_shape.n = 1;
  tl_scale_shape.c = input_c;
  tl_scale_shape.h = 1;
  tl_scale_shape.w = 1;
  tl_scale.start_address = scale_laddr;
  tl_scale.fmt = CVK_FMT_I8;
  tl_scale.shape = tl_scale_shape;
  tl_scale.stride = ctx.tl_default_stride(tl_scale_shape, CVK_FMT_I8, 1);

  // unsupport bias
  CviBackendContext::QuantizeMode qmode;
  if (i8_multiplier != nullptr) {
    qmode = CviBackendContext::QuantizeMode::INT8_PER_LAYER;
  }
  else {
    qmode = CviBackendContext::QuantizeMode::INT8_32_MULTIPLER;
  }

  cvk_tl_t tl_bias;
  cvk_tl_shape_t tl_bias_shape;
  if (qmode == CviBackendContext::QuantizeMode::INT8_32_MULTIPLER) {
    tl_bias_shape.n = 1;
    tl_bias_shape.c = input_c;
    tl_bias_shape.h = 1;
    tl_bias_shape.w = ctx.chan_quan_param_size(do_bias);
  } else if (qmode == CviBackendContext::QuantizeMode::INT8_PER_LAYER) {
    if (do_bias) {
      tl_bias_shape.n = 2;
      tl_bias_shape.c = input_c;
      tl_bias_shape.h = 1;
      tl_bias_shape.w = 1;
    }
  }
  tl_bias.start_address = bias_laddr;
  tl_bias.fmt = CVK_FMT_I8;
  tl_bias.shape = tl_bias_shape;
  tl_bias.stride = ctx.tl_default_stride(tl_bias_shape, CVK_FMT_I8, 1);

  if (qmode == CviBackendContext::QuantizeMode::INT8_PER_LAYER) {
    cvk_tiu_mul_param_t p = {0};
    p.res_high = nullptr;
    p.res_low = &tl_scale;
    p.a = &tl_scale;
    p.b_is_const = 1;
    p.b_const.val = i8_multiplier[0];
    p.b_const.is_signed = true;
    p.rshift_bits = 0;
    p.layer_id = layer_id;
    p.relu_enable = 0;
    ctx.tiu_mul(&p);
  }

  /*
   * Res(n, c, h, w) = A(n, c, h, w) * B(1,c,1,1) + Bias(1,c,1,1)
   * Use depthwise-conv to implement linear-arithmatic MAC
   * (channel-wise mac).
   */
  if (qmode == CviBackendContext::QuantizeMode::INT8_PER_LAYER) {
    cvk_tiu_depthwise_pt_convolution_param_t param = {0};
    param.ofmap = &tl_output;
    param.ifmap = &tl_input;
    param.weight = &tl_scale;
    param.bias = do_bias ? &tl_bias : nullptr;
    param.ins_h = 0;
    param.ins_last_h = 0;
    param.ins_w = 0;
    param.ins_last_w = 0;
    param.pad_top = 0;
    param.pad_bottom = 0;
    param.pad_left = 0;
    param.pad_right = 0;
    param.stride_h = 1;
    param.stride_w = 1;
    param.dilation_h = 1;
    param.dilation_w = 1;
    param.rshift_bits = right_shift_width;
    param.relu_enable = fused_relu;
    param.layer_id = layer_id;
    param.ins_val = 0;                            // symmetric quantization
    param.ins_fp = ctx.convert_fp32_to_bf16(0.0); // symmetric quantization
    ctx.tiu_pt_depthwise_convolution(&param);
  }
  else if (qmode == CviBackendContext::QuantizeMode::INT8_32_MULTIPLER) {
    tl_bias.shape = ctx.tl_shape_t4(1, input_c, 1, 1);
    tl_bias.stride = ctx.tl_default_stride(
        tl_bias.shape, CVK_FMT_I8, /*eu_aign=*/0);

    cvk_tiu_depthwise_convolution_param_t param = {nullptr};
    param.ofmap = &tl_output;
    param.ifmap = &tl_input;
    param.weight = &tl_scale;
    param.chl_quan_param = &tl_bias;
    param.ins_h = 0;
    param.ins_last_h = 0;
    param.ins_w = 0;
    param.ins_last_w = 0;
    param.pad_top = 0;
    param.pad_bottom = 0;
    param.pad_left = 0;
    param.pad_right = 0;
    param.stride_h = 1;
    param.stride_w = 1;
    param.dilation_h = 1;
    param.dilation_w = 1;
    param.has_bias = do_bias ? 1 : 0;
    param.relu_enable = fused_relu;
    param.layer_id = layer_id;
    param.ins_val = 0;                            // symmetric quantization
    param.ins_fp = ctx.convert_fp32_to_bf16(0.0); // symmetric quantization
    ctx.tiu_depthwise_convolution(&param);
  }
}


void cvi_backend_bf16_tl_broadcast_mul(
      const CviBackendContext &ctx, uint32_t layer_id,
      laddr_t input_laddr, laddr_t scale_laddr,
      laddr_t bias_laddr, laddr_t output_laddr, int input_n,
      int input_c, int input_h, int input_w,
      int do_activation,
      int activation_method,
      float activation_arg[],
      bool do_bias) {
  LLVM_DEBUG(llvm::errs() << llvm::format(
      "cvi_backend_bf16_tl_broadcast_mul:\n"
      "input_laddr 0x%lx scale_laddr 0x%lx, "
      "bias_laddr:0x%lx, output_laddr:0x%lx, "
      "input_n %d input_c %d input_h %d input_w %d "
      "\n",
      input_laddr, scale_laddr, bias_laddr, output_laddr, input_n, input_c,
      input_h, input_w););

  // output
  cvk_tl_shape_t tl_output_shape = ctx.tl_shape_t4(input_n,input_c,input_h,input_w);
  cvk_tl_t tl_output;
  tl_output.start_address = output_laddr;
  tl_output.fmt = CVK_FMT_BF16;
  tl_output.shape = tl_output_shape;
  tl_output.stride = ctx.tl_default_stride(tl_output_shape, CVK_FMT_BF16, 1);


#define RELU (0)
  bool fused_relu = (do_activation && activation_method == RELU &&
                    (activation_arg[0] == 0.0f));
  LLVM_DEBUG(llvm::errs() << "fused_relu is " << fused_relu;);

  // input
  cvk_tl_shape_t tl_input_shape = ctx.tl_shape_t4(input_n,input_c,input_h,input_w);
  cvk_tl_t tl_input;
  tl_input.start_address = input_laddr;
  tl_input.fmt = CVK_FMT_BF16;
  tl_input.shape = tl_input_shape;
  tl_input.stride = ctx.tl_default_stride(tl_input_shape, CVK_FMT_BF16, 1);

  // scale
  cvk_tl_shape_t tl_scale_shape;
  cvk_tl_t tl_scale;
  tl_scale_shape.n = 1;
  tl_scale_shape.c = input_c;
  tl_scale_shape.h = 1;
  tl_scale_shape.w = 1;
  tl_scale.start_address = scale_laddr;
  tl_scale.fmt = CVK_FMT_BF16;
  tl_scale.shape = tl_scale_shape;
  tl_scale.stride = ctx.tl_default_stride(tl_scale_shape, CVK_FMT_BF16, 1);

  cvk_tl_t tl_bias;
  cvk_tl_shape_t tl_bias_shape;
  tl_bias_shape.n = 1;
  tl_bias_shape.c = input_c;
  tl_bias_shape.h = 1;
  tl_bias_shape.w = 1;
  tl_bias.start_address = bias_laddr;
  tl_bias.fmt = CVK_FMT_BF16;
  tl_bias.shape = tl_bias_shape;
  tl_bias.stride = ctx.tl_default_stride(tl_bias_shape, CVK_FMT_BF16, 1);

  /*
   * Res(n, c, h, w) = A(n, c, h, w) * B(1,c,1,1) + Bias(1,c,1,1)
   * Use depthwise-conv to implement linear-arithmatic MAC
   * (channel-wise mac).
   */
  cvk_tiu_depthwise_pt_convolution_param_t param = {0};
  param.ofmap = &tl_output;
  param.ifmap = &tl_input;
  param.weight = &tl_scale;
  param.bias = do_bias ? &tl_bias : nullptr;
  param.ins_h = 0;
  param.ins_last_h = 0;
  param.ins_w = 0;
  param.ins_last_w = 0;
  param.pad_top = 0;
  param.pad_bottom = 0;
  param.pad_left = 0;
  param.pad_right = 0;
  param.stride_h = 1;
  param.stride_w = 1;
  param.dilation_h = 1;
  param.dilation_w = 1;
  param.relu_enable = fused_relu;
  param.layer_id = layer_id;
  param.ins_val = 0;                            // symmetric quantization
  param.ins_fp = ctx.convert_fp32_to_bf16(0.0); // symmetric quantization
  ctx.tiu_pt_depthwise_convolution(&param);
}