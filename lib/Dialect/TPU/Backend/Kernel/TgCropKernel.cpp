/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgCropKernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <cmath>

#define DEBUG_TYPE "kernel_crop"

void cvi_backend_tg_crop_kernel(const CviBackendContext &ctx, uint32_t layer_id,
                                gaddr_t ga_input, gaddr_t ga_output,
                                int *input_dim, int *output_dim, int *offsets,
                                cvk_fmt_t fmt) {
  ctx.set_layer_id(layer_id);
  cvk_tg_shape_t i_s =
      ctx.tg_shape_t4(input_dim[0], input_dim[1], input_dim[2], input_dim[3]);
  cvk_tg_stride_t i_gstride = ctx.tg_default_stride(i_s, fmt);
  cvk_tg_shape_t o_s = ctx.tg_shape_t4(output_dim[0], output_dim[1],
                                       output_dim[2], output_dim[3]);
  cvk_tg_stride_t o_gstride = ctx.tg_default_stride(o_s, fmt);

  std::vector<CviBackendContext::tiling_info_t> tiles;
  ctx.tiling_packing(tiles, o_s, fmt);
  auto max_shape =
      ctx.tl_shape_t4(tiles[0].n, tiles[0].c, tiles[0].h, tiles[0].w);
  cvk_tl_t *tl_input = ctx.lmem_alloc_tensor(max_shape, fmt, 1);
  for (auto &tile : tiles) {
    auto i_tile = tile;
    i_tile.pos_n += offsets[0];
    i_tile.pos_c += offsets[1];
    i_tile.pos_h += offsets[2];
    i_tile.pos_w += offsets[3];
    uint64_t input_offset =
        i_tile.pos_n * i_gstride.n + i_tile.pos_c * i_gstride.c +
        i_tile.pos_h * i_gstride.h + i_tile.pos_w * i_gstride.w;
    cvk_tl_t tl_ifmap = *tl_input;
    tl_ifmap.shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
    tl_ifmap.stride = ctx.tl_default_stride(tl_ifmap.shape, fmt, 1);
    ctx.tdma_load_stride(&tl_ifmap, ga_input + input_offset, i_gstride);
    ctx.tdma_store_stride(&tl_ifmap, ga_output + tile.offset, o_gstride);
  }
  ctx.lmem_free_tensor(tl_input);
}