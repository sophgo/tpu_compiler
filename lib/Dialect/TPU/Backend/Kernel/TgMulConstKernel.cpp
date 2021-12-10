/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * refined 2020-10-12
 */

#include "TgMulConstKernel.hpp"
#include "backend/backend_tl_api.h"

#define DEBUG_TYPE "TgMulConstKernel"

void TgMulConstKernel::init(uint32_t layer_id, gaddr_t ga_input,
                            gaddr_t ga_output, int32_t n, int32_t c, int32_t h,
                            int32_t w, bool do_relu, float const_val,
                            cvk_fmt_t fmt) {
  this->layer_id = layer_id;
  this->ga_input = ga_input;
  this->ga_output = ga_output;
  this->n = n;
  this->c = c;
  this->h = h;
  this->w = w;
  this->do_relu = do_relu;
  this->const_val = const_val;
  this->fmt = fmt;
  this->fmt_size = ctx.bytesize_of_fmt(fmt);
}

void TgMulConstKernel::allocLmem() {
  auto &tile = tiles[0];
  auto shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  tl_mem[0] = ctx.lmem_alloc_tensor(shape, fmt, 1);
  tl_mem[1] = ctx.lmem_alloc_tensor(shape, fmt, 1);
}

void TgMulConstKernel::deallocLmem() {
  ctx.lmem_free_tensor(tl_mem[1]);
  ctx.lmem_free_tensor(tl_mem[0]);
}

void TgMulConstKernel::selectTilePolicy() {
  ctx.tiling_packing(tiles, ctx.tg_shape_t4(n, c, h, w), fmt, 2, 0,
                     CviBackendContext::TilingAll);
}

void TgMulConstKernel::refresh(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  cvk_tl_shape_t shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  ctx.lmem_init_tensor(&tl_input, shape, fmt, 1);
  tl_input.start_address = tl_mem[step_idx % 2]->start_address;
}

void TgMulConstKernel::load(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  ctx.tdma_load(&tl_input, ga_input + tile.offset);
}

void TgMulConstKernel::store(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  ctx.tdma_store(&tl_input, ga_output + tile.offset);
}

void TgMulConstKernel::compute(int32_t step_idx) {
  if (const_val == 1.0f && do_relu == false) {
    return;
  }
  refresh(step_idx);
  if (fmt == CVK_FMT_I8) {
    cvk_tiu_mul_param_t p1 = {0};
    p1.res_high = nullptr;
    p1.res_low = &tl_input;
    p1.a = &tl_input;
    p1.b_const.val = static_cast<int16_t>(const_val);
    p1.b_const.is_signed = true;
    p1.b_is_const = 1;
    p1.rshift_bits = 0;
    p1.layer_id = layer_id;
    p1.relu_enable = do_relu;
    ctx.tiu_mul(&p1);
  } else {
    cvk_tiu_mul_param_t p1 = {0};
    p1.res_high = nullptr;
    p1.res_low = &tl_input;
    p1.a = &tl_input;
    p1.b_const.val = ctx.convert_fp32_to_bf16(const_val);
    p1.b_const.is_signed = 1;
    p1.b_is_const = 1;
    p1.rshift_bits = 0;
    p1.layer_id = layer_id;
    p1.relu_enable = do_relu;
    ctx.tiu_mul(&p1);
  }
}

void TgMulConstKernel::schedule() {
  allocLmem();
  int32_t total_steps = tiles.size();
  for (int32_t i = 0; i < total_steps + 2; ++i) {
    ctx.parallel_enable();
    if (i - 1 >= 0 && i - 1 < total_steps) {
      compute(i - 1);
    }
    if (i - 2 >= 0) {
      store(i - 2);
    }
    if (i < total_steps) {
      load(i);
    }
    ctx.parallel_disable();
  }
  deallocLmem();
}

void cvi_backend_tg_mul_const_kernel(const CviBackendContext &ctx,
                                     uint32_t layer_id, gaddr_t ga_input,
                                     gaddr_t ga_output, int32_t n, int32_t c,
                                     int32_t h, int32_t w, bool do_relu,
                                     float const_val, cvk_fmt_t fmt) {
  TgMulConstKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_output, n, c, h, w, do_relu, const_val,
              fmt);
  kernel.selectTilePolicy();
  kernel.schedule();
}
