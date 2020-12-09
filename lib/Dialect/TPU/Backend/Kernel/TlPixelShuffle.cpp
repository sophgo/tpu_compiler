/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TlPixelShuffle.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include "backend/backend_tl_api.h"

#define DEBUG_TYPE "tl_pixel_shuffle"

static void pixelShuffle_tensor_load_nc_transpose(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_ifmap,
    cvk_tg_stride_t &ifmap_gstride, laddr_t ifmap_laddr, int n_pos, int c_pos,
    cvk_fmt_t fmt, cvk_tl_shape_t tileShape) {
  uint64_t ga_ifmap_offset = ifmap_gstride.n * n_pos + ifmap_gstride.c * c_pos;

  cvk_tg_t tg_ifmap = {0};
  tg_ifmap.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_ifmap);
  tg_ifmap.fmt = fmt;
  tg_ifmap.start_address = ga_ifmap + ga_ifmap_offset;
  tg_ifmap.shape = {
      tileShape.n, tileShape.c, tileShape.h,
      tileShape.w};
  tg_ifmap.stride =  ifmap_gstride;
  tg_ifmap.stride.n = tg_ifmap.stride.c * tileShape.c;

  cvk_tl_t tl_dst;
  tl_dst.start_address = ifmap_laddr;  // Same as ifmap = 0
  tl_dst.fmt = fmt;
  tl_dst.shape = {tileShape.c, tileShape.n, tileShape.h, tileShape.w};
  tl_dst.stride = ctx.tl_default_stride(tl_dst.shape, fmt, /*eu_align=*/0);

  cvk_tdma_g2l_tensor_copy_nc_transposed_param_t param = {0};
  param.src = &tg_ifmap;
  param.dst = &tl_dst;
  param.layer_id = layer_id;

  LLVM_DEBUG(llvm::dbgs()
      << "  pixelShuffle_tensor_load_nc_transpose\n"
      << "    tg offset " << ga_ifmap_offset
      << ", shape(" << param.src->shape.n
      << ", " << param.src->shape.c << ", " << param.src->shape.h
      << "), stride(" << param.src->stride.n
      << ", " << param.src->stride.c << ", " << param.src->stride.h << ")\n"
      << "    tl shape(" << param.dst->shape.n
      << ", " << param.dst->shape.c << ", " << param.dst->shape.h
      << ", " << param.dst->shape.w
      << "), stride(" << param.dst->stride.n
      << ", " << param.dst->stride.c << ", " << param.dst->stride.h
      << ", " << param.dst->stride.w << ")\n");

  ctx.tdma_g2l_tensor_copy_nc_transposed(&param);
}

static void pixelShuffle_tiu_copy(
            const CviBackendContext &ctx, uint32_t layer_id, cvk_tl_t *tl_ifmap,
            cvk_tl_t *tl_ofmap, int factor) {

  cvk_tl_t tl_dst;
  tl_dst.start_address = tl_ofmap->start_address;  // start of lmem
  tl_dst.fmt = tl_ofmap->fmt;
  tl_dst.shape = tl_ofmap->shape;
  int bytesize = tl_ofmap->stride.w;
  tl_dst.stride = {
      (uint32_t)(factor * tl_ofmap->shape.w * bytesize),
      (uint32_t)bytesize,
      (uint32_t)(factor * factor * tl_ifmap->shape.w * bytesize),
      (uint32_t)(factor * bytesize)
  };

  cvk_tiu_copy_param_t p2 = {0};
  p2.src = tl_ifmap;
  p2.dst = &tl_dst;
  p2.layer_id = layer_id;

  LLVM_DEBUG(llvm::errs() << llvm::format(
    "    [%d] L2L Reshape:\n"
    "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n"
    "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n",
    p2.src->start_address, p2.src->shape.n,
    p2.src->shape.c, p2.src->shape.h, p2.src->shape.w, p2.src->stride.n,
    p2.src->stride.c, p2.src->stride.h, p2.src->stride.w, p2.dst->start_address,
    p2.dst->shape.n, p2.dst->shape.c, p2.dst->shape.h, p2.dst->shape.w,
    p2.dst->stride.n, p2.dst->stride.c, p2.dst->stride.h, p2.dst->stride.w));

  ctx.tiu_copy(&p2);
}

void cvi_backend_tl_pixel_shuffle_LA(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t ifmap_laddr, laddr_t ofmap_laddr, gaddr_t ga_ifmap,
    uint32_t input_n, uint32_t input_c, uint32_t input_h, uint32_t input_w,
    uint32_t factor) {

  uint32_t in = input_n;
  uint32_t ic = input_c;
  uint32_t ih = input_h;
  uint32_t iw = input_w;
  uint32_t oc = input_c / (factor * factor);
  uint32_t oh = input_h * factor;
  uint32_t ow = input_w * factor;

  int eu_align = 0; // no need to align eu
  cvk_fmt_t fmt = CVK_FMT_I8;

  uint32_t n_step = 1;
  uint32_t oc_step = (oc >= (uint32_t)NPU_NUM) ? NPU_NUM : oc;
  uint32_t c_step = oc_step * factor * factor;

  cvk_tg_stride_t ifmap_gstride = { ic * ih * iw, ih * iw, iw};
  //cvk_tg_stride_t ofmap_gstride = { oc * oh * ow, oh * ow, ow};
  laddr_t tile_laddr =  ofmap_laddr;

  for (uint32_t n_pos = 0; n_pos < in; n_pos += n_step) {
    for (uint32_t c_pos = 0; c_pos < ic; c_pos += c_step) {
      uint32_t tiling_c = std::min(ic - c_pos, c_step);
      // 1. Assign local memory layout
      cvk_tl_t tl_ifmap, tl_ofmap;

      ctx.lmem_init_tensor(&tl_ifmap,
                  {factor, factor * NPU_NUM, ih, iw}, fmt, eu_align);
      tl_ifmap.start_address = ifmap_laddr;

      ctx.lmem_init_tensor(&tl_ofmap,
                  {factor, factor * NPU_NUM, ih, iw}, fmt, eu_align);
      tl_ofmap.start_address = tile_laddr;

      // 2. tensor load
      cvk_tl_shape_t tileShape = {oc_step, factor * factor, input_h, iw};
      pixelShuffle_tensor_load_nc_transpose(ctx, layer_id, ga_ifmap,
                     ifmap_gstride, ifmap_laddr, n_pos, c_pos, fmt, tileShape);

      // 3. tiu copy
      pixelShuffle_tiu_copy(ctx, layer_id, &tl_ifmap, &tl_ofmap, factor);

      // 4. tensor store
      //uint32_t oc_pos = c_pos / (factor * factor);
      cvk_tl_shape_t outputTileShape = {1, tiling_c / (factor * factor),
                                        oh, ow};
      tile_laddr += ctx.lmem_tensor_to_size(outputTileShape, fmt, eu_align);
    }
  }
}
