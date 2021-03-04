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

#define DEBUG_TYPE "cvi_backend_scalelut_kernel"

void TgScaleLutKernel::init(uint32_t layer_id, gaddr_t ga_input,
                            gaddr_t ga_output, gaddr_t ga_lut, int n, int c,
                            int h, int w, cvk_fmt_t fmt) {
  this->layer_id = layer_id;
  this->n = n;
  this->c = c;
  this->h = h;
  this->w = w;
  this->c_times = 1;
  this->ga_input = ga_input;
  this->ga_output = ga_output;
  this->ga_lut = ga_lut;
  this->fmt = fmt;
  reshape();
  lut_shape = ctx.lut_table_shape(fmt);
  gstride = ctx.tg_default_stride(this->c, this->h, this->w, fmt);
  ctx.set_layer_id(layer_id);
}

// for example
// 1 3 540 960 => 1 30 1 54*960 = 1 30 180 288
#define MAX_H_STRIDE 0x10000
void TgScaleLutKernel::reshape() {
  int c_ = c;
  int h_ = 1;
  int w_ = h * w;
  int max_times = NPU_NUM / c;
  for (int time = max_times; time >= 2; time--) {
    if (w_ % time == 0) {
      w_ /= time;
      c_ *= time;
      break;
    }
  }
  if (c == c_) {
    return;
  }
  if (w_ > MAX_WIDTH) {
    int div = std::sqrt(w_);
    for (int time = div; time >= 2; time--) {
      if (w_ % time == 0) {
        w_ /= time;
        h_ = time;
        break;
      }
    }
    if (w_ >= MAX_H_STRIDE && h_ >= MAX_H_STRIDE) {
      return;
    }
    if (w_ >= MAX_H_STRIDE) {
      std::swap(w_, h_);
    } else if (h_ > w_ && h_ < MAX_H_STRIDE) {
      std::swap(w_, h_);
    }
  }
  LLVM_DEBUG(
      llvm::dbgs() << llvm::format("reshape [%d,%d,%d,%d] => [%d,%d,%d,%d]\n",
                                   n, c, h, w, n, c_, h_, w_));
  c_times = c_ / c;
  c = c_;
  h = h_;
  w = w_;
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
  // load table
  int hw = lut_shape.h * lut_shape.w;
  cvk_tg_t ts_data = {0};
  ts_data.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(ga_lut);
  ts_data.fmt = fmt;
  ts_data.start_address = ga_lut;
  ts_data.shape = ctx.tg_shape_t4(1, c / c_times, c_times, hw);
  ts_data.stride = {(uint32_t)(c / c_times * hw), (uint32_t)hw, 0, 1};
  tl_lut->shape = ctx.tl_shape_t4(1, c, 1, hw);
  tl_lut->stride = ctx.tl_default_stride(tl_lut->shape, fmt, 1);
  cvk_tdma_g2l_tensor_copy_param_t p = {0};
  p.src = &ts_data;
  p.dst = tl_lut;
  p.layer_id = layer_id;
  ctx.tdma_g2l_tensor_copy(&p);
  tl_lut->shape = lut_shape;
  tl_lut->stride = ctx.tl_default_stride(lut_shape, fmt, 1);
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