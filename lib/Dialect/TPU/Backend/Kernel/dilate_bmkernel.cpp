/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: dilate_bmkernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <cmath>

#define DEBUG_TYPE "bmnet_bm1880_bmkernel_dilate"
#define DEBUG_SPLIT "bmnet_bm1880_bmkernel_dilate_split"

void cvi_backend_tg_fixed_dilate_kernel(const CviBackendContext &ctx,
                                 uint32_t stream_id, uint32_t inst_id, uint32_t layer_id,
                                 const uint32_t *depends, uint32_t depends_len,
                                 gaddr_t bottom_gaddr, gaddr_t top_gaddr,
                                 int n, int c, int ih, int iw,
                                 int oh, int ow, int fill_constant,
                                 int ins_h, int ins_w,
                                 cvk_fmt_t _fmt) {

  cvk_fmt_t fmt = cvi_to_cvk_fmt(_fmt);

  int data_size = (fmt == CVK_FMT_BF16) ? sizeof(uint16_t) : sizeof(uint8_t);
  assert((fmt == CVK_FMT_BF16 || fmt == CVK_FMT_I8 || fmt == CVK_FMT_U8) &&
      "dilate ONLY support bf16/i8");

  cvk_tg_shape_t input_shape = {(uint32_t)n, (uint32_t)c, (uint32_t)ih, (uint32_t)iw};
  cvk_tg_shape_t output_shape = {(uint32_t)n, (uint32_t)c, (uint32_t)oh, (uint32_t)ow};
  cvk_tg_stride_t input_gstride, output_gstride;

  // g2g with stride
  // 1. fill target with 0
  // 2. fill following settings
  // src: nc h w 1 -> nc h w 1
  // src_sh,sc,sn = 1, w, hw
  // dst_sh,sc,sn = ins_w + 1, ow * (ins_h+1), ow*oh

  cvk_tdma_l2g_tensor_fill_constant_param_t p = {0};
  cvk_tg_t dst;
  dst.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(top_gaddr);
  dst.start_address = top_gaddr;
  dst.fmt = fmt;
  dst.shape = {
    static_cast<uint32_t>(n),
    static_cast<uint32_t>(c),
    static_cast<uint32_t>(oh),
    static_cast<uint32_t>(ow)
  };
  dst.stride = ctx.tg_default_stride(dst.shape, dst.fmt);

  p.constant = fill_constant;
  p.dst = &dst;
  ctx.tdma_l2g_tensor_fill_constant(&p);

  // NOTICE: make sure tdma order before than mul
  ctx.parallel_enable();
  ctx.parallel_disable();

  // 2. sys->sys
  input_shape.n = n * c;
  input_shape.c = ih;
  input_shape.h = iw;
  input_shape.w = 1;

  input_gstride.h = 1 * data_size;
  input_gstride.c = iw * data_size;
  input_gstride.n = ih * iw * data_size;

  output_shape = input_shape;
  output_gstride.h = (ins_w + 1) * data_size;
  output_gstride.c = ow * (ins_h+1) * data_size;
  output_gstride.n = ow*oh * data_size;

  tdma_g2g_tensor_copy(
      ctx,
      bottom_gaddr, input_shape, input_gstride, fmt,
      top_gaddr, output_shape, output_gstride, fmt
      );
}
