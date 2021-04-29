/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgPadKernel.cpp
 * Description:
 */

#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include "CviBackendContext.h"

#define DEBUG_TYPE "cvi_backend_pad_kernel"

// input shape (in, ic, ih, iw)
// output_shape (on, oc, oh, ow)
// pads (x0_begin, x1_begin, x2_begin, x3_begin, x0_end, x1_end, x2_end, x3_end)
//
// on = x0_begin + x0_end + in
// oc = x1_begin + x1_end + ic
// oh = x2_begin + x2_end + ih
// ow = x3_begin + x3_end + iw

void cvi_backend_tg_pad_kernel(
    const CviBackendContext& ctx,
    uint32_t layer_id, gaddr_t ga_ifmap, gaddr_t ga_ofmap, int input_n,
    int input_c, int input_h, int input_w, int *pads, float const_val,
    cvk_fmt_t fmt) {

  ctx.set_layer_id(layer_id);

  cvk_tg_shape_t src_shape;
  src_shape.n = input_n;
  src_shape.c = input_c;
  src_shape.h = input_h;
  src_shape.w = input_w;
  int usize = (fmt == CVK_FMT_I8) ? 1 : 2;

  cvk_tg_stride_t src_gstride;
  src_gstride = ctx.tg_default_stride(src_shape, fmt);

  cvk_tg_shape_t dst_shape;
  dst_shape.n = pads[0] + pads[4] + input_n;
  dst_shape.c = pads[1] + pads[5] + input_c;
  dst_shape.h = pads[2] + pads[6] + input_h;
  dst_shape.w = pads[3] + pads[7] + input_w;

  cvk_tg_stride_t dst_gstride;
  dst_gstride = ctx.tg_default_stride(dst_shape, fmt);

  cvk_tg_t dst;
  dst.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_ofmap);
  dst.int8_rnd_mode = 0;
  dst.fmt = fmt;
  dst.start_address = ga_ofmap;
  dst.shape = dst_shape;
  dst.stride = ctx.tg_default_stride(dst.shape, dst.fmt);

  cvk_tdma_l2g_tensor_fill_constant_param_t p0;
  if (fmt == CVK_FMT_BF16) {
    p0.constant = ctx.convert_fp32_to_bf16(const_val);
  } else if (fmt == CVK_FMT_I8) {
    assert(const_val >= -128 && const_val <= 127);
    int8_t val = (int8_t)const_val;
    p0.constant = *((uint8_t *)&val);
  } else {
    assert(0);
  }
  p0.dst = &dst;
  ctx.tdma_l2g_tensor_fill_constant(&p0);

  auto src_gaddr = ga_ifmap;
  auto dst_gaddr = ga_ofmap + (dst_shape.w * pads[2] + pads[3]) * usize;
  ctx.tdma_g2g_tensor_copy(src_gaddr, src_shape, src_gstride, fmt, dst_gaddr,
                           src_shape, dst_gstride, fmt);
}
