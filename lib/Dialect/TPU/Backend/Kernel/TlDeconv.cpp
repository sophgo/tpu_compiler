/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: tl_deconv.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include "backend/backend_tl_api.h"

#define DEBUG_TYPE "tl_deconv"

static void dump(cvk_tiu_convolution_param_t &param, int ifmap_offset,
                 int weight_offset, int ofmap_offset) {
  LLVM_DEBUG(llvm::errs() << llvm::format("ins_h: %d, ins_last_h: %d, "
                  "ins_w: %d, ins_last_w: %d, pad_top: %d, pad_bottom: %d, "
                  "pad_left: %d, pad_right: %d, dilation_h: %d, "
                  "dilation_w: %d\n", param.ins_h,  param.ins_last_h,
                  param.ins_w, param.ins_last_w, param.pad_top,
                  param.pad_bottom, param.pad_left, param.pad_right,
                  param.dilation_h, param.dilation_w););

  LLVM_DEBUG(llvm::errs() << llvm::format("input_offset:%d, "
                            "input shape: (%d, %d, %d, %d), "
                            "input stride:(%d, %d, %d, 1)\n",
                            ifmap_offset,
                            param.ifmap->shape.n, param.ifmap->shape.c,
                            param.ifmap->shape.h, param.ifmap->shape.w,
                            param.ifmap->stride.n,
                            param.ifmap->stride.c, param.ifmap->stride.h););

  LLVM_DEBUG(llvm::errs() << llvm::format("weight_offset:%d, "
                          "weight shape: (%d, %d, %d, %d), "
                          "weight stride:(%d, %d, %d, 1)\n",
                          weight_offset,
                          param.weight->shape.n, param.weight->shape.c,
                          param.weight->shape.h, param.weight->shape.w,
                          param.weight->stride.n,
                          param.weight->stride.c, param.weight->stride.h););

  LLVM_DEBUG(llvm::errs() << llvm::format("output_offset:%d, "
                            "output shape: (%d, %d, %d, %d), "
                            "output stride:(%d, %d, %d, 1)\n\n",
                            ofmap_offset,
                            param.ofmap->shape.n, param.ofmap->shape.c,
                            param.ofmap->shape.h, param.ofmap->shape.w,
                            param.ofmap->stride.n,
                            param.ofmap->stride.c, param.ofmap->stride.h););
}

void cvi_backend_tl_depthwise_deconv(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_ifmap, laddr_t la_ofmap, laddr_t la_weight,
    laddr_t la_perchannel,
    int input_n, int input_c, int input_h, int input_w, int group,
    int output_c, int output_h, int output_w,
    int kh, int kw, int dh, int dw,
    int ins_h, int ins_last_h, int ins_w, int ins_last_w,
    int pad_h_top, int pad_h_bottom, int pad_w_left, int pad_w_right,
    int stride_h, int stride_w, bool do_bias,
    bool do_relu,
    int rshift, int rshift_len) {

  cvk_tl_t tl_input;
  cvk_tl_t tl_output;
  cvk_tl_t tl_weight;

  // input
  cvk_tl_shape_t tl_input_shape = ctx.tl_shape_t4(input_n,input_c,input_h,input_w);
  tl_input.start_address = la_ifmap;
  tl_input.fmt = CVK_FMT_I8;
  tl_input.shape = tl_input_shape;
  tl_input.stride = ctx.tl_default_stride(tl_input_shape, CVK_FMT_I8, 1);

// output
  cvk_tl_shape_t tl_output_shape = ctx.tl_shape_t4(input_n,output_c,output_h,output_w);
  tl_output.start_address = la_ofmap;
  tl_output.fmt = CVK_FMT_I8;
  tl_output.shape = tl_output_shape;
  tl_output.stride = ctx.tl_default_stride(tl_output_shape, CVK_FMT_I8, 1);

  // weight
  cvk_tl_shape_t tl_weight_shape = ctx.tl_shape_t4(1,output_c,kh,kw);
  tl_weight.start_address = la_weight;
  tl_weight.fmt = CVK_FMT_I8;
  tl_weight.shape = tl_weight_shape;
  tl_weight.stride = ctx.tl_default_stride(tl_weight_shape, CVK_FMT_I8, 1);

  cvk_tl_t tl_chl_quant = {0};
  tl_chl_quant.start_address = la_perchannel;
  tl_chl_quant.fmt = CVK_FMT_I8;
  tl_chl_quant.shape = {1, (unsigned int)output_c, 1, 1};
  tl_chl_quant.stride =
      ctx.tl_default_stride(tl_chl_quant.shape, CVK_FMT_I8, /*eu_align=*/0);

  cvk_tiu_depthwise_convolution_param_t param = {0};
  memset(&param, 0, sizeof(param));
  param.ofmap = &tl_output;
  param.ifmap = &tl_input;
  param.weight = &tl_weight;
  param.chl_quan_param = &tl_chl_quant;
  param.ins_h = ins_h;
  param.ins_last_h = ins_last_h;
  param.ins_w = ins_w;
  param.ins_last_w = ins_last_w;
  param.stride_h = 1;
  param.stride_w = 1;
  param.dilation_h = dh;
  param.dilation_w = dw;
  param.pad_top = pad_h_top;
  param.pad_bottom = pad_h_bottom;
  param.pad_left = pad_w_left;
  param.pad_right = pad_w_right;
  param.has_bias = do_bias;
  param.relu_enable = do_relu;
  ctx.tiu_depthwise_convolution(&param);
}


void cvi_backend_tl_deconv(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_ifmap, laddr_t la_ofmap, laddr_t la_weight,
    laddr_t la_perchannel,
    int input_n, int input_c, int input_h, int input_w, int group,
    int output_c, int output_h, int output_w,
    int kh, int kw, int dh, int dw,
    int ins_h, int ins_last_h, int ins_w, int ins_last_w,
    int pad_h_top, int pad_h_bottom, int pad_w_left, int pad_w_right,
    int stride_h, int stride_w, bool do_bias,
    bool do_relu,
    int rshift, int rshift_len,
    bool do_ic_alignment) {

  LLVM_DEBUG(llvm::errs() << llvm::format(
                  "cvi_backend_tl_deconv:\n"
                  "    layer_id %d\n"
                  "    la_ifmap 0x%lx, la_ofmap_0x%lx, la_weight 0x%lx, la_perchannel 0x%lx\n"
                  "    in(%d, %d, %d, %d), out(,%d, %d, %d), group %d kernel(%d,%d)\n"
                  "    pad(%d, %d, %d, %d), stride(%d, %d)\n"
                  "    ins_h %d ins_w %d ins_last_h %d ins_last_w %d"
                  "    rshift %d, do_bias %d, do_relu %d\n"
                  "    rshift_len %d\n",
                  layer_id, la_ifmap, la_ofmap, la_weight, la_perchannel,
                  input_n, input_c, input_h, input_w, output_c, output_h, output_w,
                  group, kh, kw, pad_h_top, pad_h_bottom, pad_w_left,
                  pad_w_right, stride_h, stride_w,
                  ins_h, ins_w, ins_last_h, ins_last_w,
                  rshift, do_bias, do_relu,
                  rshift_len));

  if (do_ic_alignment && (input_c % 2 != 0)) {
    input_c = input_c + 1;
  }

  bool do_chl_quan = rshift_len ? true : false;
  // only support chl_quan now.
  assert(do_chl_quan == true);

  if (input_c == group && output_c == group && group != 1) {
    cvi_backend_tl_depthwise_deconv(
        ctx, layer_id,
        la_ifmap, la_ofmap, la_weight, la_perchannel,
        input_n, input_c, input_h, input_w, group,
        output_c, output_h, output_w,
        kh, kw, dh, dw, ins_h, ins_last_h, ins_w, ins_last_w,
        pad_h_top, pad_h_bottom, pad_w_left, pad_w_right,
        stride_h, stride_w, do_bias, do_relu,
        rshift, rshift_len);
    return;
  }

  cvk_tl_t tl_input;
  cvk_tl_t tl_output;
  cvk_tl_t tl_weight;

  int ic = input_c / group;
  int oc = output_c / group;

  for (int ig = 0; ig < group; ig++) {
    // input
    auto ifmap_offset = ((ic * ig) / NPU_NUM) *
                        align_up(input_h * input_w, EU_NUM) +
                        ((ic * ig) % NPU_NUM) * LOCAL_MEM_SIZE;

    cvk_tl_shape_t tl_input_shape = ctx.tl_shape_t4(input_n,ic,input_h,input_w);
    tl_input.start_address = la_ifmap + ifmap_offset;
    tl_input.fmt = CVK_FMT_I8;
    tl_input.shape = tl_input_shape;
    tl_input.stride = ctx.tl_default_stride(tl_input_shape, CVK_FMT_I8, 1);

    // output
    auto ofmap_offset = ((oc * ig) / NPU_NUM) *
                        align_up(input_h * input_w, EU_NUM) +
                        ((oc * ig) % NPU_NUM) * LOCAL_MEM_SIZE;
    cvk_tl_shape_t tl_output_shape = ctx.tl_shape_t4(input_n,oc,output_h,output_w);
    tl_output.start_address = la_ofmap + ofmap_offset;
    tl_output.fmt = CVK_FMT_I8;
    tl_output.shape = tl_output_shape;
    tl_output.stride = ctx.tl_default_stride(tl_output_shape, CVK_FMT_I8, 1);

    // weight
    auto weight_offset = ((oc * ig) / NPU_NUM) * ic * input_h * input_w +
                        ((oc * ig) % NPU_NUM) * LOCAL_MEM_SIZE;
    cvk_tl_shape_t tl_weight_shape = ctx.tl_shape_t4(ic,oc,kh,kw);
    tl_weight.start_address = la_weight + weight_offset;
    tl_weight.fmt = CVK_FMT_I8;
    tl_weight.shape = tl_weight_shape;
    tl_weight.stride = ctx.tl_default_stride(tl_weight_shape, CVK_FMT_I8, 1);

    cvk_tl_t tl_chl_quant = {0};
    tl_chl_quant.start_address = la_perchannel;
    tl_chl_quant.fmt = CVK_FMT_I8;
    tl_chl_quant.shape = {1, (unsigned int)oc, 1, 1};
    tl_chl_quant.stride =
        ctx.tl_default_stride(tl_chl_quant.shape, CVK_FMT_I8, /*eu_align=*/0);

    cvk_tiu_convolution_param_t param = {0};
    param.ofmap = &tl_output;
    param.ifmap = &tl_input;
    param.weight = &tl_weight;
    param.chl_quan_param = &tl_chl_quant;
    param.ins_h = ins_h;
    param.ins_last_h = ins_last_h;
    param.ins_w = ins_w;
    param.ins_last_w = ins_last_w;
    param.pad_top = pad_h_top;
    param.pad_bottom = pad_h_bottom;
    param.pad_left = pad_w_left;
    param.pad_right = pad_w_right;
    param.stride_h = 1;
    param.stride_w = 1;
    param.dilation_h = dh;
    param.dilation_w = dw;
    param.relu_enable = do_relu;
    param.ps32_mode = 0;
    param.w_is_const = 0;
    param.layer_id = layer_id;
    param.has_bias = do_bias ? 1 : 0;
    ctx.tiu_convolution(&param);
    dump(param, ifmap_offset, weight_offset, ofmap_offset);
  }
}


void cvi_backend_tl_bf16_depthwise_deconv(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_ifmap, laddr_t la_ofmap, laddr_t la_weight,
    laddr_t la_bias,
    int input_n, int input_c, int input_h, int input_w, int group,
    int output_c, int output_h, int output_w,
    int kh, int kw, int dh, int dw,
    int ins_h, int ins_last_h, int ins_w, int ins_last_w,
    int pad_h_top, int pad_h_bottom, int pad_w_left, int pad_w_right,
    int stride_h, int stride_w, bool do_bias,
    bool do_relu) {

  cvk_tl_t tl_input;
  cvk_tl_t tl_output;
  cvk_tl_t tl_weight;

  // input
  cvk_tl_shape_t tl_input_shape = ctx.tl_shape_t4(input_n,input_c,input_h,input_w);
  tl_input.start_address = la_ifmap;
  tl_input.fmt = CVK_FMT_BF16;
  tl_input.shape = tl_input_shape;
  tl_input.stride = ctx.tl_default_stride(tl_input_shape, CVK_FMT_BF16, 1);

// output
  cvk_tl_shape_t tl_output_shape = ctx.tl_shape_t4(input_n,output_c,output_h,output_w);
  tl_output.start_address = la_ofmap;
  tl_output.fmt = CVK_FMT_BF16;
  tl_output.shape = tl_output_shape;
  tl_output.stride = ctx.tl_default_stride(tl_output_shape, CVK_FMT_BF16, 1);

  // weight
  cvk_tl_shape_t tl_weight_shape = ctx.tl_shape_t4(1,output_c,kh,kw);
  tl_weight.start_address = la_weight;
  tl_weight.fmt = CVK_FMT_BF16;
  tl_weight.shape = tl_weight_shape;
  tl_weight.stride = ctx.tl_default_stride(tl_weight_shape, CVK_FMT_BF16, 1);

  cvk_tl_t tl_bias;
  tl_bias.start_address = la_bias;
  tl_bias.fmt = CVK_FMT_BF16;
  tl_bias.shape = ctx.tl_shape_t4(2, output_c, 1, 1);
  tl_bias.stride =
      ctx.tl_default_stride(tl_bias.shape, CVK_FMT_BF16, 0);

  cvk_tiu_depthwise_pt_convolution_param_t param = {0};
  memset(&param, 0, sizeof(param));
  param.ofmap = &tl_output;
  param.ifmap = &tl_input;
  param.weight = &tl_weight;
  param.bias = do_bias ? &tl_bias : 0;
  param.ins_h = ins_h;
  param.ins_last_h = ins_last_h;
  param.ins_w = ins_w;
  param.ins_last_w = ins_last_w;
  param.stride_h = 1;
  param.stride_w = 1;
  param.dilation_h = dh;
  param.dilation_w = dw;
  param.pad_top = pad_h_top;
  param.pad_bottom = pad_h_bottom;
  param.pad_left = pad_w_left;
  param.pad_right = pad_w_right;
  param.relu_enable = do_relu;
  ctx.tiu_pt_depthwise_convolution(&param);
}

void cvi_backend_tl_bf16_deconv(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_ifmap, laddr_t la_ofmap, laddr_t la_weight,
    laddr_t la_bias,
    int input_n, int input_c, int input_h, int input_w, int group,
    int output_c, int output_h, int output_w,
    int kh, int kw, int dh, int dw,
    int ins_h, int ins_last_h, int ins_w, int ins_last_w,
    int pad_h_top, int pad_h_bottom, int pad_w_left, int pad_w_right,
    int stride_h, int stride_w, bool do_bias,
    bool do_relu) {

  LLVM_DEBUG(llvm::errs() << llvm::format(
                  "cvi_backend_tl_bf16_deconv:\n"
                  "    layer_id %d\n"
                  "    la_ifmap 0x%lx, la_ofmap_0x%lx, la_weight 0x%lx, la_bias 0x%lx\n"
                  "    in(%d, %d, %d, %d), out(,%d, %d, %d), group %d kernel(%d,%d)\n"
                  "    pad(%d, %d, %d, %d), stride(%d, %d)\n"
                  "    ins_h %d ins_w %d ins_last_h %d ins_last_w %d"
                  "    do_bias %d, do_relu %d\n",
                  layer_id, la_ifmap, la_ofmap, la_weight, la_bias,
                  input_n, input_c, input_h, input_w, output_c, output_h, output_w,
                  group, kh, kw, pad_h_top, pad_h_bottom, pad_w_left,
                  pad_w_right, stride_h, stride_w,
                  ins_h, ins_w, ins_last_h, ins_last_w,
                  do_bias, do_relu));

  if (input_c == group && output_c == group && group != 1) {
    cvi_backend_tl_bf16_depthwise_deconv(
        ctx, layer_id,
        la_ifmap, la_ofmap, la_weight, la_bias,
        input_n, input_c, input_h, input_w, group,
        output_c, output_h, output_w,
        kh, kw, dh, dw, ins_h, ins_last_h, ins_w, ins_last_w,
        pad_h_top, pad_h_bottom, pad_w_left, pad_w_right,
        stride_h, stride_w, do_bias, do_relu);
    return;
  }

  cvk_tl_t tl_input;
  cvk_tl_t tl_output;
  cvk_tl_t tl_weight;

  int ic = input_c / group;
  int oc = output_c / group;

  for (int ig = 0; ig < group; ig++) {
    // input
    auto ifmap_offset = ((ic * ig) / NPU_NUM) *
                        align_up(input_h * input_w * 2, EU_NUM) +
                        ((ic * ig) % NPU_NUM) * LOCAL_MEM_SIZE;

    cvk_tl_shape_t tl_input_shape = ctx.tl_shape_t4(input_n,ic,input_h,input_w);
    tl_input.start_address = la_ifmap + ifmap_offset;
    tl_input.fmt = CVK_FMT_BF16;
    tl_input.shape = tl_input_shape;
    tl_input.stride = ctx.tl_default_stride(tl_input_shape, CVK_FMT_BF16, 1);

    // output
    auto ofmap_offset = ((oc * ig) / NPU_NUM) *
                        align_up(input_h * input_w * 2, EU_NUM) +
                        ((oc * ig) % NPU_NUM) * LOCAL_MEM_SIZE;
    cvk_tl_shape_t tl_output_shape = ctx.tl_shape_t4(input_n,oc,output_h,output_w);
    tl_output.start_address = la_ofmap + ofmap_offset;
    tl_output.fmt = CVK_FMT_BF16;
    tl_output.shape = tl_output_shape;
    tl_output.stride = ctx.tl_default_stride(tl_output_shape, CVK_FMT_BF16, 1);

    // weight
    auto weight_offset = ((oc * ig) / NPU_NUM) * ic * input_h * input_w * 2+
                        ((oc * ig) % NPU_NUM) * LOCAL_MEM_SIZE;
    cvk_tl_shape_t tl_weight_shape = ctx.tl_shape_t4(ic,oc,kh,kw);
    tl_weight.start_address = la_weight + weight_offset;
    tl_weight.fmt = CVK_FMT_BF16;
    tl_weight.shape = tl_weight_shape;
    tl_weight.stride = ctx.tl_default_stride(tl_weight_shape, CVK_FMT_BF16, 1);

    cvk_tl_t tl_bias = {0};
    tl_bias.start_address = la_bias;
    tl_bias.fmt = CVK_FMT_BF16;
    tl_bias.shape = {2, (unsigned int)oc, 1, 1};
    tl_bias.stride =
        ctx.tl_default_stride(tl_bias.shape, CVK_FMT_BF16, /*eu_align=*/0);

    cvk_tiu_pt_convolution_param_t param = {0};
    param.ofmap = &tl_output;
    param.ifmap = &tl_input;
    param.weight = &tl_weight;
    param.bias = do_bias ? &tl_bias : 0;
    param.ins_h = ins_h;
    param.ins_last_h = ins_last_h;
    param.ins_w = ins_w;
    param.ins_last_w = ins_last_w;
    param.pad_top = pad_h_top;
    param.pad_bottom = pad_h_bottom;
    param.pad_left = pad_w_left;
    param.pad_right = pad_w_right;
    param.stride_h = 1;
    param.stride_w = 1;
    param.dilation_h = dh;
    param.dilation_w = dw;
    param.relu_enable = do_relu;
    param.ps32_mode = 0;
    param.w_is_const = 0;
    param.layer_id = layer_id;
    ctx.tiu_pt_convolution(&param);
  }
}