/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgScaleLutKernel.cpp
 * Description:
 */

#include "TgScaleLutKernel.hpp"
#include "CviBackendContext.h"
#include "backend/backend_tg_api.h"
#include <cmath>
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

void TgScaleLutKernel::init(uint32_t layer_id, gaddr_t ga_input,
                            gaddr_t ga_output, gaddr_t ga_lut, int n, int c,
                            int h, int w, cvk_fmt_t fmt) {
  this->layer_id = layer_id;
  this->n = n;
  this->c = c;
  this->h = h;
  this->w = w;
  this->ga_input = ga_input;
  this->ga_output = ga_output;
  this->ga_lut = ga_lut;
  this->fmt = fmt;
  lut_shape = ctx.lut_table_shape(fmt);
  gstride = ctx.tg_default_stride(c, h, w, fmt);
  ctx.set_layer_id(layer_id);
}

void TgScaleLutKernel::selectTilePolicy() {
  uint32_t lmem_used = ctx.lmem_tensor_to_size(lut_shape, fmt, 1);
  ctx.tiling_packing(tiles, n, c, h, w, fmt, BLOB_NUM, lmem_used,
                     CviBackendContext::TilingNHW);
}

void TgScaleLutKernel::allocLmem() {
  cvk_tl_shape_t gshape =
      ctx.tl_shape_t4(tiles[0].n, tiles[0].c, tiles[0].h, tiles[0].w);
  tl_lut = ctx.lmem_alloc_tensor(lut_shape, fmt, 1);
  for (int i = 0; i < BLOB_NUM; i++) {
    tl_mem[i] = ctx.lmem_alloc_tensor(gshape, fmt, 1);
  }

  ctx.tdma_load(tl_lut, ga_lut);
}

void TgScaleLutKernel::deallocLmem() {
  for (int i = BLOB_NUM - 1; i >= 0; i--) {
    ctx.lmem_free_tensor(tl_mem[i]);
  }
  ctx.lmem_free_tensor(tl_lut);
}

void TgScaleLutKernel::schedule() {
  allocLmem();
  int32_t total_steps = tiles.size();
  for (int32_t i = 0; i < total_steps + 2; i++) {
    ctx.parallel_enable();

    if (i - 1 >= 0 && i - 1 < total_steps) {
      compute(i - 1);
    }
    if (i < total_steps) {
      load(i);
    }
    if (i - 2 >= 0) {
      store(i - 2);
    }
    ctx.parallel_disable();
  }
  deallocLmem();
}

void TgScaleLutKernel::refresh(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  tl_ifmap = *tl_mem[step_idx % 2];
  tl_ofmap = *tl_mem[2 + step_idx % 2];
  auto shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  auto stride = ctx.tl_default_stride(shape, fmt, 1);
  tl_ifmap.shape = shape;
  tl_ifmap.stride = stride;
  tl_ofmap.shape = shape;
  tl_ofmap.stride = stride;
}

void TgScaleLutKernel::load(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  ctx.tdma_load_stride(&tl_ifmap, ga_input + tile.offset, gstride);
}

void TgScaleLutKernel::store(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  ctx.tdma_store_stride(&tl_ofmap, ga_output + tile.offset, gstride);
}

void TgScaleLutKernel::compute(int32_t step_idx) {
  refresh(step_idx);
  cvk_tiu_lookup_table_param_t p = {0};
  p.ofmap = &tl_ofmap;
  p.ifmap = &tl_ifmap;
  p.table = tl_lut;
  p.layer_id = layer_id;
  ctx.tiu_lookup_table(&p);
}

void cvi_backend_tg_scale_lut_kernel(const CviBackendContext &ctx,
                                     uint32_t layer_id, gaddr_t ga_input,
                                     gaddr_t ga_output, gaddr_t ga_lut, int n,
                                     int c, int h, int w, cvk_fmt_t fmt) {
  TgScaleLutKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_output, ga_lut, n, c, h, w, fmt);
  kernel.selectTilePolicy();
  kernel.schedule();
}