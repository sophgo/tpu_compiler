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

static void cvi_backend_tg_pad_kernel_edge(const CviBackendContext& ctx,
    uint32_t layer_id, gaddr_t ga_ifmap, gaddr_t ga_ofmap, int input_n,
    int input_c, int input_h, int input_w, int *pads, cvk_fmt_t fmt) {
  // pad left and right
  // pad top
  // pad bottom
  assert(pads[0] == pads[4] && pads[1] == pads[5] && pads[0] == 0 && pads[1] == 0
      && "only support h/w pad");
  assert(input_n + pads[0] + pads[4] == 1 && "not support n slice");

  cvk_tg_shape_t dst_shape;
  dst_shape.n = pads[0] + pads[4] + input_n;
  dst_shape.c = pads[1] + pads[5] + input_c;
  dst_shape.h = pads[2] + pads[6] + input_h;
  dst_shape.w = pads[3] + pads[7] + input_w;
  cvk_tg_stride_t dst_gstride;
  dst_gstride = ctx.tg_default_stride(dst_shape, fmt);

  cvk_tg_shape_t src_shape;
  src_shape.n = input_n;
  src_shape.c = input_c;
  src_shape.h = input_h;
  src_shape.w = input_w;
  cvk_tg_stride_t src_gstride;
  src_gstride = ctx.tg_default_stride(src_shape, fmt);

  int blob_num = 1;
  std::vector<CviBackendContext::tiling_info_t> tiles;
  // NOTICE: only tile h
  ctx.tiling_packing(tiles, input_n, input_c, src_shape.h, dst_shape.w, fmt, blob_num,
      /*reserved_lmem=*/0, CviBackendContext::TilingNH);

  // 1. load
  // 2.1 pad w by pads[3], pads[7]
  // 2.2 handle h = 0 case and duplicate by pads[2]
  // 2.4 handle h = ow case and duplicate by pads[6]
  // 3. write
  bool eu_align = false;
  int store_off = 0;
  int load_off = 0;

  // prepare lmem
  auto &tile = tiles[0];
  cvk_tl_shape_t in_tl_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);

  cvk_tl_t *tl_ifmap = ctx.lmem_alloc_tensor(in_tl_shape, fmt, eu_align);
  auto tl_ifmap_addr = tl_ifmap->start_address;

  for (int i = 0; i < (int)tiles.size(); i++) {
    auto &tile = tiles[i];

    // reshape for each tile
    in_tl_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, input_w);
    tl_ifmap->shape = in_tl_shape;

    in_tl_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
    auto in_tl_stride = ctx.tl_default_stride(in_tl_shape, fmt, eu_align);
    tl_ifmap->stride = in_tl_stride;

    // copy after pads[3]
    tl_ifmap->start_address = tl_ifmap_addr + pads[3] * tl_ifmap->stride.w;
    ctx.tdma_load_stride(tl_ifmap, ga_ifmap + load_off, src_gstride);
    load_off += tl_ifmap->shape.h * tl_ifmap->shape.w * tl_ifmap->stride.w;

    cvk_tiu_copy_param_t param = {0};
    cvk_tl_t tl_ofmap, _tl_ifmap;

    if (pads[3]) {
      tl_ofmap = *tl_ifmap;
      _tl_ifmap = *tl_ifmap;
      tl_ofmap.start_address = tl_ifmap_addr;
      tl_ofmap.shape.w = pads[3];
      tl_ofmap.stride.h = dst_shape.w * tl_ofmap.stride.w;
      tl_ofmap.stride.c = tile.h * tl_ofmap.stride.h;
      tl_ofmap.stride.n = 0; // n MUST eq 1

      _tl_ifmap.stride.w = 0;
      _tl_ifmap.shape = tl_ofmap.shape;

      // duplicate pads[3]
      param.src = &_tl_ifmap;
      param.dst = &tl_ofmap;
      param.layer_id = layer_id;
      ctx.tiu_copy(&param);
    }

    if (pads[7]) {
      tl_ofmap = *tl_ifmap;
      _tl_ifmap = *tl_ifmap;
      tl_ofmap.start_address = _tl_ifmap.start_address + (src_shape.w * tl_ofmap.stride.w);
      tl_ofmap.shape.w = pads[7];
      tl_ofmap.stride.h = dst_shape.w * tl_ofmap.stride.w;
      tl_ofmap.stride.c = tile.h * tl_ofmap.stride.h;
      tl_ofmap.stride.n = 0; // n MUST eq 1

      _tl_ifmap.start_address = _tl_ifmap.start_address +
        (src_shape.w - 1) * _tl_ifmap.stride.w;
      _tl_ifmap.stride.w = 0;
      _tl_ifmap.shape = tl_ofmap.shape;

      // duplicate pads[7]
      param.src = &_tl_ifmap;
      param.dst = &tl_ofmap;
      param.layer_id = layer_id;
      ctx.tiu_copy(&param);
    }

    tl_ifmap->start_address = tl_ifmap_addr;
    if (store_off == 0 && pads[2]) {
      // top
      tl_ofmap = *tl_ifmap;
      tl_ofmap.shape.w = dst_shape.w;
      tl_ofmap.stride = ctx.tl_default_stride(tl_ofmap.shape, fmt, eu_align);
      tl_ofmap.shape.h = pads[2];
      tl_ofmap.stride.h = 0;
      ctx.tdma_store_stride(&tl_ofmap, ga_ofmap + store_off, dst_gstride);
      store_off += tl_ofmap.shape.w * pads[2] * tl_ofmap.stride.w;
    }

    in_tl_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
    in_tl_stride = ctx.tl_default_stride(in_tl_shape, fmt, eu_align);
    tl_ifmap->shape = in_tl_shape;
    tl_ifmap->stride = in_tl_stride;

    ctx.tdma_store_stride(tl_ifmap, ga_ofmap + store_off, dst_gstride);
    store_off += tile.h * tile.w * tl_ifmap->stride.w;
  }

  if (pads[6]) {
    cvk_tl_t tl_ofmap;
    // shift to last
    tl_ifmap->start_address = tl_ifmap_addr +
      tl_ifmap->shape.w * (tl_ifmap->shape.h - 1) * tl_ifmap->stride.w;

    tl_ofmap = *tl_ifmap;
    tl_ofmap.stride = ctx.tl_default_stride(tl_ofmap.shape, fmt, eu_align);
    tl_ofmap.shape.h = pads[6];
    tl_ofmap.stride.h = 0;

    ctx.tdma_store_stride(&tl_ofmap, ga_ofmap + store_off, dst_gstride);
    store_off += tl_ifmap->stride.c;
  }

  // release
  in_tl_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  tl_ifmap->shape = in_tl_shape; // free by shape size
  tl_ifmap->start_address = tl_ifmap_addr;
  ctx.lmem_free_tensor(tl_ifmap);
}

void cvi_backend_tg_pad_kernel(
    const CviBackendContext& ctx,
    uint32_t layer_id, gaddr_t ga_ifmap, gaddr_t ga_ofmap, int input_n,
    int input_c, int input_h, int input_w, int *pads, float const_val,
    const char* mode, cvk_fmt_t fmt) {

  ctx.set_layer_id(layer_id);

  if (!strcmp(mode, "edge")) {
    return cvi_backend_tg_pad_kernel_edge(ctx,
        layer_id, ga_ifmap, ga_ofmap, input_n,
        input_c, input_h, input_w, pads, fmt);
  }

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
