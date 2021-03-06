/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgFixedDilateKernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <cmath>

#define DEBUG_TYPE "bm1880_kernel_dilate"
#define DEBUG_SPLIT "bm1880_kernel_dilate_split"

void cvi_backend_tg_fixed_dilate_kernel(const CviBackendContext &ctx,
                                        uint32_t layer_id, gaddr_t bottom_gaddr,
                                        gaddr_t top_gaddr, int n, int c, int ih,
                                        int iw, int oh, int ow,
                                        int fill_constant, int ins_h, int ins_w,
                                        cvk_fmt_t fmt) {
  ctx.assert_support_fmt(fmt);
  int data_size = ctx.bytesize_of_fmt(fmt);

  cvk_tg_shape_t input_shape = ctx.tg_shape_t4(n,c,ih,iw);
  cvk_tg_shape_t output_shape = ctx.tg_shape_t4(n,c,oh,ow);
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
  dst.shape = ctx.tg_shape_t4(n,c,oh,ow);
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

  ctx.tdma_g2g_tensor_copy(bottom_gaddr, input_shape, input_gstride, fmt, top_gaddr,
                           output_shape, output_gstride, fmt);
}
