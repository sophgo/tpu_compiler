/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 */

#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <cmath>
#include <iostream>
#include "CviBackendContext.h"

#define DEBUG_TYPE "TgBf16Pooling3dKernel"

// Input
// global memory (n, ic, id, ih, iw)
// local memory  (n*id, ic, ih, iw), align eu
static void loadInput(
    const CviBackendContext &ctx,
    int n, int ic, int id, int ih, int iw,
    gaddr_t ga_input, cvk_tl_t *tl_input_al)
{
  // local memory layout (id, ic, ih, iw)
  cvk_fmt_t fmt = tl_input_al->fmt;
  cvk_tl_shape_t tl_shape = ctx.tl_shape_t4(1, ic, id, ih * iw);
  cvk_tl_t tl_input;
  ctx.lmem_init_tensor(&tl_input, tl_shape, fmt, tl_input_al->eu_align);
  tl_input.start_address = tl_input_al->start_address;
  tl_input.stride.h = llvm::alignTo(tl_input.stride.h, EU_NUM);
  tl_input.stride.n = tl_input.stride.h * id;

  // global memory (1, ic, id, ih*iw)
  cvk_tg_shape_t tg_shape = ctx.tg_shape_t4(1, ic, id, ih * iw);
  cvk_tg_t gm_input;
  ctx.gmem_init_tensor(&gm_input, tg_shape, fmt);
  gm_input.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_input);
  gm_input.start_address = ga_input;

  cvk_tdma_g2l_tensor_copy_param_t param = {0};
  param.src = &gm_input;
  param.dst = &tl_input;
  ctx.tdma_g2l_tensor_copy(&param);
}

static void compute(
    const CviBackendContext &ctx,
    int n, int ic, int id, int ih, int iw,
    int od, int oh, int ow,
    int pad_d0, int pad_d1,
    int pad_top, int pad_bot, int pad_left, int pad_right,
    int kd, int kh, int kw,
    int stride_d, int stride_h, int stride_w,
    cvk_tl_t *tl_input_al, cvk_tl_t *tl_work, cvk_tl_t *tl_output_al)
{
  (void)pad_d1;
  (void)pad_bot;
  (void)pad_right;
  assert(n == 1 && "Only support batch 1");

  cvk_fmt_t fmt = tl_input_al->fmt;

  // Apply 2d max pool to each input depth
  {
    cvk_tl_shape_t input_shape = ctx.tl_shape_t4(id, ic, ih, iw);
    cvk_tl_t tl_input;
    ctx.lmem_init_tensor(&tl_input, input_shape, fmt, /*eu_align=*/1);
    tl_input.start_address = tl_input_al->start_address;

    cvk_tl_shape_t output_shape = ctx.tl_shape_t4(id, ic, oh, ow);
    cvk_tl_t tl_output;
    ctx.lmem_init_tensor(&tl_output, output_shape, fmt, /*eu_align=*/1);
    tl_output.start_address = tl_output_al->start_address;

    cvk_tiu_max_pooling_param_t param = {0};
    param.ofmap = &tl_output;
    param.ifmap = &tl_input;
    param.kh = kh;
    param.kw = kw;
    param.pad_top = (uint8_t)pad_top;
    param.pad_bottom = (uint8_t)pad_bot;
    param.pad_left = (uint8_t)pad_left;
    param.pad_right = (uint8_t)pad_right;
    param.stride_h = (uint8_t)stride_h;
    param.stride_w = (uint8_t)stride_w;
    param.ins_val = -128;
    param.ins_fp = 0xff7f;
    ctx.tiu_max_pooling(&param);
  }

  // TIU copy (od, ic, oh, ow) -> (1, ic, od, oh*ow)
  // from eu-aligned to contiguous
  {
    cvk_tl_shape_t tl_input_shape = ctx.tl_shape_t4(1, ic, oh, ow);

    cvk_tl_stride_t aligned_stride =
        ctx.tl_default_stride(tl_input_shape, fmt, 1);
    cvk_tl_stride_t unaligned_stride =
        ctx.tl_default_stride(tl_input_shape, fmt, 0);

    for (int oz = 0; oz < od; ++oz) {
      for (int pd = 0; pd < kd; pd++) {
        int iz = oz * stride_d + pd - pad_d0;

        cvk_tl_t tl_input;
        ctx.lmem_init_tensor(&tl_input, tl_input_shape, fmt, /*eu_align=*/1);
        tl_input.start_address = tl_output_al->start_address +
                                 aligned_stride.n * iz;

        cvk_tl_shape_t tl_output_shape = {
            1, (uint32_t)ic, (uint32_t)oh, (uint32_t)ow};
        cvk_tl_t tl_output;
        ctx.lmem_init_tensor(&tl_output, tl_output_shape, fmt, /*eu_align=*/0);
        tl_output.start_address = tl_work->start_address +
                                  aligned_stride.n * oz * kd +
                                  unaligned_stride.n * pd;

        cvk_tiu_copy_param_t param = {0};
        param.src = &tl_input;
        param.dst = &tl_output;
        ctx.tiu_copy(&param);
      }
    }
  }

  // Apply 2d max pool to input depth
  // input (od, ic, kd, oh*ow)
  // kernel (id, 1)
  // output (od, ic, 1, oh*ow)
  {
    cvk_tl_shape_t tiu_copy_input_shape = ctx.tl_shape_t4(1, ic, oh, ow);

    cvk_tl_stride_t tiu_copy_aligned_stride =
        ctx.tl_default_stride(tiu_copy_input_shape, fmt, 1);

    for (int oz = 0; oz < od; ++oz) {
      cvk_tl_shape_t tl_input_shape = ctx.tl_shape_t4(1, ic, kd, oh * ow);
      cvk_tl_t tl_input;
      ctx.lmem_init_tensor(&tl_input, tl_input_shape, fmt, /*eu_align=*/1);
      tl_input.start_address = tl_work->start_address +
                               tiu_copy_aligned_stride.n * oz * kd;

      cvk_tl_shape_t tl_output_shape = ctx.tl_shape_t4(1, ic, 1, oh * ow);
      cvk_tl_t tl_output;
      ctx.lmem_init_tensor(&tl_output, tl_output_shape, fmt, /*eu_align=*/1);
      tl_output.start_address = tl_output_al->start_address +
                                tl_output.stride.n * oz;

      cvk_tiu_max_pooling_param_t param = {0};
      param.ofmap = &tl_output;
      param.ifmap = &tl_input;
      param.kh = kd;
      param.kw = 1;
      param.pad_top = 0;
      param.pad_bottom = 0;
      param.pad_left = 0;
      param.pad_right = 0;
      param.stride_h = 1;
      param.stride_w = 1;
      param.ins_val = -128;
      param.ins_fp = 0xff7f;
      ctx.tiu_max_pooling(&param);
    }
  }
}

static void storeOutput(
    const CviBackendContext &ctx,
    int n, int ic, int od, int oh, int ow,
    gaddr_t ga_output, cvk_tl_t *tl_output_al) {
  assert(n == 1 && "Only support batch 1");

  cvk_fmt_t fmt = tl_output_al->fmt;
  cvk_tl_shape_t tl_output_shape = ctx.tl_shape_t4(1, ic, od, oh * ow);
  cvk_tl_t tl_output;
  ctx.lmem_init_tensor(&tl_output, tl_output_shape, fmt, /*eu_align=*/1);
  tl_output.start_address = tl_output_al->start_address;
  tl_output.stride.h = llvm::alignTo(tl_output.stride.h, EU_NUM);

  cvk_tg_shape_t tg_output_shape = ctx.tg_shape_t4(1, ic, od, oh * ow);
  cvk_tg_t tg_output = {0};
  ctx.gmem_init_tensor(&tg_output, tg_output_shape, fmt);
  tg_output.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_output);
  tg_output.start_address = ga_output;

  cvk_tdma_l2g_tensor_copy_param_t param = {0};
  param.src = &tl_output;
  param.dst = &tg_output;
  ctx.tdma_l2g_tensor_copy(&param);
}

void cvi_backend_tg_bf16_max_pooling3d_kernel(
    const CviBackendContext &ctx,
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
    int input_n, int input_c, int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kd, int kh, int kw,
    int pad_d0, int pad_d1,
    int pad_top, int pad_bot, int pad_left, int pad_right,
    int stride_d, int stride_h, int stride_w,
    bool do_relu, const bool ceil_mode) {
  assert(input_n == 1 && pad_d0 == 0 && pad_d1 == 0 &&
         "Only support batch 1, no depth pad");
  assert(!do_relu && "Not support relu");

  cvk_fmt_t fmt = CVK_FMT_BF16;
  cvk_tl_t *tl_input_al = NULL;
  cvk_tl_t *tl_output_al = NULL;
  cvk_tl_t *tl_work_al = NULL;

  // Allocate input (n*id, ic, ih, iw)
  // treat input depth as batch, do 2d max pool for each input depth
  // align EU for tiu max pool.
  cvk_tl_shape_t tl_input_al_shape =
      ctx.tl_shape_t4(input_n * input_d, input_c, input_h, input_w);
  tl_input_al = ctx.lmem_alloc_tensor(tl_input_al_shape, fmt, /*eu_align=*/1);

  // Allocate work (n=od*kd, ic, oh, ow)
  cvk_tl_shape_t tl_work_al_shape =
      ctx.tl_shape_t4(output_d * kd, input_c, output_h, output_w);
  tl_work_al = ctx.lmem_alloc_tensor(tl_work_al_shape, fmt, /*eu_align=*/1);

  // Allocate output (n=1, ic, od, oh*ow)
  cvk_tl_shape_t tl_output_al_shape =
      ctx.tl_shape_t4(input_n, input_c, output_d, output_h * output_w);
  tl_output_al = ctx.lmem_alloc_tensor(tl_output_al_shape, fmt, /*eu_align=*/0);

  // load input
  loadInput(ctx,
            input_n, input_c, input_d, input_h, input_w,
            ga_input, tl_input_al);

  // 3d max pool
  compute(ctx,
          input_n, input_c, input_d, input_h, input_w,
          output_d, output_h, output_w,
          pad_d0, pad_d1,
          pad_top, pad_bot, pad_left, pad_right,
          kd, kh, kw,
          stride_d, stride_h, stride_w,
          tl_input_al, tl_work_al, tl_output_al);

  // store output
  storeOutput(ctx,
              input_n, input_c, output_d, output_h, output_w,
              ga_output, tl_output_al);

  // Reverse order
  ctx.lmem_free_tensor(tl_output_al);
  ctx.lmem_free_tensor(tl_work_al);
  ctx.lmem_free_tensor(tl_input_al);
}
