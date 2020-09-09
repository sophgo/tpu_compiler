/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: tl_tdma.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include "backend/backend_tl_api.h"

#define DEBUG_TYPE "tl_tdma"

void cvi_backend_tl_load(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_ifmap, gaddr_t ga_ifmap,
    uint32_t n, uint32_t ic, uint32_t ih, uint32_t iw) {
  cvk_tl_t tl_ifmap;
  tl_ifmap.start_address = la_ifmap;
  tl_ifmap.fmt = CVK_FMT_I8;
  tl_ifmap.shape = ctx.shape_t4(n, ic, ih, iw);
  tl_ifmap.stride = ctx.tl_default_stride(tl_ifmap.shape, CVK_FMT_I8, /*eu_align=*/1);

  cvk_tg_stride_t ifmap_gstride = {ic * ih * iw, ih * iw, iw};
  ctx.set_layer_id(layer_id);
  ctx.tdma_load_stride(&tl_ifmap, ga_ifmap, ifmap_gstride);
}

void cvi_backend_tl_store(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_ofmap, gaddr_t ga_ofmap,
    uint32_t n, uint32_t oc, uint32_t oh, uint32_t ow) {
  cvk_tl_t tl_ofmap;
  tl_ofmap.start_address = la_ofmap;
  tl_ofmap.fmt = CVK_FMT_I8;
  tl_ofmap.shape = ctx.shape_t4(n, oc, oh, ow);
  tl_ofmap.stride = ctx.tl_default_stride(tl_ofmap.shape, CVK_FMT_I8, /*eu_align=*/1);

  cvk_tg_stride_t ofmap_gstride = {oc * oh * ow, oh * ow, ow};
  ctx.set_layer_id(layer_id);
  ctx.tdma_store_stride(&tl_ofmap, ga_ofmap, ofmap_gstride);
}

void cvi_backend_tl_copy(
  const CviBackendContext &ctx, uint32_t layer_id,
  int la_src, int la_dst,
  int n, int c, int h, int w,
  bool align) {

  ctx.set_layer_id(layer_id);

  cvk_tl_t tl_dst;
  cvk_tl_t tl_src;

  tl_src.start_address = la_src;
  tl_src.fmt = CVK_FMT_I8;
  tl_src.shape = ctx.shape_t4(n, c, h, w);
  tl_src.stride = ctx.tl_default_stride(tl_src.shape, CVK_FMT_I8, align);

  tl_dst.start_address = la_dst;
  tl_dst.fmt = CVK_FMT_I8;
  tl_dst.shape = ctx.shape_t4(n, c, h, w);
  tl_dst.stride = ctx.tl_default_stride(tl_dst.shape, CVK_FMT_I8, align);

  // tl_copy.dst = &tl_dst;
  // tl_copy.src = &tl_src;
  // ctx.tdma_l2l_tensor_copy(&tl_copy);

  cvk_tiu_mul_param_t p2 = {0};
  p2.res_high = nullptr;
  p2.res_low = &tl_dst;
  p2.a = &tl_src;
  p2.b_const.val = 1;
  p2.b_const.is_signed = true;
  p2.b_is_const = 1;
  p2.rshift_bits = 0;
  p2.layer_id = layer_id;
  p2.relu_enable = 0;
  ctx.tiu_mul(&p2);
}
