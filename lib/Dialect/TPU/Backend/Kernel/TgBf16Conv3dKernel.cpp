/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgBf16Conv3dKernel.cpp
 * Description:
 *
 */

#include "CviBackendContext.h"
#include "backend/backend_tl_api.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>

#define DEBUG_TYPE "tg_bf16_conv3d_kernel"

static void loadBias(const CviBackendContext &ctx, uint64_t ga_bias,
    cvk_tl_t *tl_bias_al)
{
  cvk_fmt_t fmt = tl_bias_al->fmt;
  cvk_tg_shape_t gm_bias_shape = {
      tl_bias_al->shape.n, tl_bias_al->shape.c, tl_bias_al->shape.h,
      tl_bias_al->shape.w};
  cvk_tg_t gm_bias;
  ctx.gmem_init_tensor(&gm_bias, gm_bias_shape, fmt);
  gm_bias.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_bias);
  gm_bias.start_address = ga_bias;

  cvk_tdma_g2l_tensor_copy_param_t param = {0};
  param.src = &gm_bias;
  param.dst = tl_bias_al;
  ctx.tdma_g2l_tensor_copy(&param);
}

// Input (n, ic, id, ih, iw)
static void loadInput(const CviBackendContext &ctx, uint32_t layer_id,
    int n, int ic, int id, int ih, int iw, int idi, uint64_t ga_input,
    cvk_tl_t *tl_input_al)
{
  // reshape (n, ic, id, ih, iw) => (n, ic, id, ih*iw)
  cvk_fmt_t fmt = tl_input_al->fmt;
  cvk_tl_shape_t tl_shape = ctx.tl_shape_t4(n, ic, 1, ih * iw);
  cvk_tl_t tl_input;
  ctx.lmem_init_tensor(&tl_input, tl_shape, fmt, tl_input_al->eu_align);
  tl_input.start_address = tl_input_al->start_address;

  uint32_t ds = (fmt == CVK_FMT_BF16) ? 2 : 1;
  cvk_tg_shape_t gm_input_shape = ctx.tg_shape_t4(n, ic, 1, ih * iw);
  cvk_tg_stride_t gm_input_stride = {ic*id*ih*iw*ds, id*ih*iw*ds, ih*iw*ds, ds};

  cvk_tg_t gm_input;
  ctx.gmem_init_tensor(&gm_input, gm_input_shape, fmt);
  gm_input.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_input);
  gm_input.start_address = ga_input + gm_input_stride.h * idi;
  gm_input.stride = gm_input_stride;

  cvk_tdma_g2l_tensor_copy_param_t param = {0};
  param.src = &gm_input;
  param.dst = &tl_input;
  ctx.tdma_g2l_tensor_copy(&param);
}

// TPU weight (kd, oc, kh*kw, ic)
static void loadWeight(const CviBackendContext &ctx, int oc, int ic, int kd,
    int kh, int kw, uint64_t ga_weight, cvk_tl_t *tl_weight_al)
{
  cvk_fmt_t fmt = tl_weight_al->fmt;
  cvk_tg_shape_t gm_weight_shape = ctx.tg_shape_t4(kd, oc, kh * kw, ic);
  cvk_tg_t gm_weight;
  ctx.gmem_init_tensor(&gm_weight, gm_weight_shape, fmt);
  gm_weight.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_weight);
  gm_weight.start_address = ga_weight;

  cvk_tdma_g2l_tensor_copy_param_t param = {0};
  param.src = &gm_weight;
  param.dst = tl_weight_al;
  ctx.tdma_g2l_tensor_copy(&param);
}

void initWeightForTiu(const CviBackendContext &ctx, cvk_tl_t *tl_weight,
    uint32_t la_weight, int oc, int ic, int kh, int kw, int kdi,
    cvk_fmt_t fmt) {
  cvk_tl_shape_t shape = ctx.tl_shape_t4(1, oc, kh * kw, ic);
  ctx.lmem_init_tensor(tl_weight, shape, fmt, /*eu_align=*/0);
  tl_weight->start_address = la_weight + tl_weight->stride.n * kdi;
}

static int get_ps32_mode(int kdi, int kd) {
  if (kd == 1)
    return 0;

  if (kdi == 0)
    return 2; // [1]: write
  else if (kdi == (kd - 1))
    return 1; // [0]: read

  return 3; // [1]: write, [0]: read
}

static void compute(const CviBackendContext &ctx, int n, int ic,
    int kh, int kw, int oc, int oh, int ow, int ps32_mode,
    cvk_tl_t *tl_input_al, cvk_tl_t *tl_weight_al, cvk_tl_t *tl_bias_al,
    cvk_tl_t *tl_output_al)
{
  cvk_fmt_t fmt = tl_weight_al->fmt;
  cvk_tl_shape_t tl_output_shape = ctx.tl_shape_t4(n, oc, oh, ow);
  cvk_tl_t tl_output;
  ctx.lmem_init_tensor(&tl_output, tl_output_shape, fmt, /*eu_align=*/1);
  tl_output.start_address = tl_output_al->start_address;
  cvk_tl_t *tl_input = tl_input_al;
  cvk_tl_shape_t tl_weight_shape = ctx.tl_shape_t4(ic, oc, kh, kw);
  cvk_tl_t tl_weight;
  ctx.lmem_init_tensor(&tl_weight, tl_weight_shape, fmt, /*eu_align=*/0);
  tl_weight.start_address = tl_weight_al->start_address;

  cvk_tl_shape_t tl_bias_shape = ctx.tl_shape_t4(2, oc, 1, 1);
  cvk_tl_t tl_bias;
  if (tl_bias_al) {
    ctx.lmem_init_tensor(&tl_bias, tl_bias_shape, fmt, /*eu_align=*/0);
    tl_bias.start_address = tl_bias_al->start_address;
  }

  cvk_tiu_pt_convolution_param_t param = {0};
  param.ifmap = tl_input;
  param.ofmap = &tl_output;
  param.weight = &tl_weight;
  param.bias = (tl_bias_al && ps32_mode == 1) ? &tl_bias : NULL;
  param.stride_h = 1;
  param.stride_w = 1;
  param.dilation_h = 1;
  param.dilation_w = 1;
  param.ps32_mode = ps32_mode;
  param.ins_val = 0;                            // symmetric quantization
  param.ins_fp = ctx.convert_fp32_to_bf16(0.0); // symmetric quantization
  ctx.tiu_pt_convolution(&param);
}

static void storeOutput(const CviBackendContext &ctx, int oc, int od, int oh,
    int ow, int odi, uint64_t ga_res, cvk_tl_t *tl_res)
{
  cvk_fmt_t fmt = tl_res->fmt;
  uint32_t ds = (fmt == CVK_FMT_BF16) ? 2 : 1;

  // Global memory shape (n, oc, od, oh, ow)
  cvk_tg_shape_t tg_res_shape = {
      tl_res->shape.n, tl_res->shape.c, tl_res->shape.h, tl_res->shape.w};
  cvk_tg_stride_t tg_stride = {
      oc * od * oh * ow * ds, od * oh * ow * ds, ow * ds, ds};
  uint32_t od_stride = oh * ow * ds;

  cvk_tg_t gm_res;
  ctx.gmem_init_tensor(&gm_res, tg_res_shape, fmt);
  gm_res.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_res);
  gm_res.start_address = ga_res + od_stride * odi;
  gm_res.stride = tg_stride;

  cvk_tdma_l2g_tensor_copy_param_t param;
  memset(&param, 0, sizeof(param));
  param.src = tl_res;
  param.dst = &gm_res;
  ctx.tdma_l2g_tensor_copy(&param);
}

void cvi_backend_tg_bf16_conv3d_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t ga_ifmap, gaddr_t ga_ofmap, gaddr_t ga_weight, gaddr_t ga_bias,
    int input_n, int input_c, int input_d, int input_h, int input_w,
    int output_c, int output_d, int output_h, int output_w,
    uint16_t kd, uint16_t kh, uint16_t kw,
    uint16_t dilation_d, uint16_t dilation_h, uint16_t dilation_w,
    uint8_t pad_d0, uint8_t pad_d1,
    uint8_t pad_top, uint8_t pad_bottom, uint8_t pad_left, uint8_t pad_right,
    uint8_t stride_d, uint8_t stride_h, uint8_t stride_w,
    bool has_bias, bool do_relu) {

  assert(input_n == 1 && "Only support batch 1");
  assert(!pad_d0 && !pad_d1 && "Not support depth padding");
  assert(stride_d == 1 && "Only support stride_d 1");

  cvk_fmt_t fmt = CVK_FMT_BF16;
  cvk_tl_shape_t tl_output_shape =
      ctx.tl_shape_t4(4 * input_n, output_c, output_h, output_w);
  cvk_tl_shape_t tl_input_shape =
      ctx.tl_shape_t4(input_n, input_c, input_h, input_w);
  cvk_tl_shape_t tl_weight_shape =
      ctx.tl_shape_t4(kd, output_c, kh*kw, input_c);
  cvk_tl_shape_t tl_bias_shape = ctx.tl_shape_t4(2, output_c, 1, 1);

  cvk_tl_t *tl_output = ctx.lmem_alloc_tensor(tl_output_shape, fmt,
                                              /*eu_align=*/1);
  cvk_tl_t *tl_input = ctx.lmem_alloc_tensor(tl_input_shape, fmt,
                                             /*eu_align=*/1);
  cvk_tl_t *tl_weight = ctx.lmem_alloc_tensor(tl_weight_shape, fmt,
                                              /*eu_align=*/0);
  cvk_tl_t *tl_bias = nullptr;
  if (has_bias)
    tl_bias = ctx.lmem_alloc_tensor(tl_bias_shape, fmt, /*eu_align=*/0);

  assert(tl_output && tl_input && tl_weight && "Expect all allocated");

  if (has_bias)
    loadBias(ctx, ga_bias, tl_bias);

  loadWeight(ctx, output_c, input_c, kd, kh, kw, ga_weight, tl_weight);

  for (int odi = 0; odi < output_d; ++odi) {
    int id_start = odi; // not support padding

    for (int kdi = 0; kdi < kd; ++kdi) {
      int idi = id_start + kdi;
      int ps32_mode = get_ps32_mode(kdi, kd);
      cvk_tl_t tl_weight_tiu;
      initWeightForTiu(ctx, &tl_weight_tiu, tl_weight->start_address,
                       output_c, input_c, kh, kw, kdi, fmt);

      loadInput(ctx, layer_id, input_n, input_c, input_d, input_h, input_w, idi,
                ga_ifmap, tl_input);
      compute(ctx, input_n, input_c, kh, kw, output_c, output_h, output_w,
              ps32_mode, tl_input, &tl_weight_tiu, tl_bias, tl_output);
    }
    storeOutput(ctx, output_c, output_d, output_h, output_w, odi, ga_ofmap,
                tl_output);
  }

  if (tl_bias)
    ctx.lmem_free_tensor(tl_bias);
  ctx.lmem_free_tensor(tl_weight);
  ctx.lmem_free_tensor(tl_input);
  ctx.lmem_free_tensor(tl_output);
}
