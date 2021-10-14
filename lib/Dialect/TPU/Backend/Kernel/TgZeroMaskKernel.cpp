/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgSliceKernel.cpp
 * Description:
 */

#include "TgZeroMaskKernel.hpp"
#include "CviBackendContext.h"
#include <cmath>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

void TgZeroMaskKernel::init(uint32_t layer_id, gaddr_t ga_input,
                            gaddr_t ga_output, int n, int c, int h, int w,
                            bool positive, cvk_fmt_t fmt) {
  this->layer_id = layer_id;
  this->n = n;
  this->c = c;
  this->h = h;
  this->w = w;
  this->ga_input = ga_input;
  this->ga_output = ga_output;
  this->positive = positive;
  this->fmt = fmt;
  ctx.set_layer_id(layer_id);
}

void TgZeroMaskKernel::selectTilePolicy() {
  ctx.tiling_packing(tiles, n, c, h, w, fmt, BLOB_NUM, 0,
                     CviBackendContext::TilingAll);
}

void TgZeroMaskKernel::allocLmem() {
  cvk_tl_shape_t shape =
      ctx.tl_shape_t4(tiles[0].n, tiles[0].c, tiles[0].h, tiles[0].w);
  for (int i = 0; i < BLOB_NUM; i++) {
    tl_mem[i] = ctx.lmem_alloc_tensor(shape, fmt, 1);
  }
}

void TgZeroMaskKernel::deallocLmem() {
  for (int i = BLOB_NUM - 1; i >= 0; i--) {
    ctx.lmem_free_tensor(tl_mem[i]);
  }
}

void TgZeroMaskKernel::refresh(int32_t step_idx) {
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

void TgZeroMaskKernel::load(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  ctx.tdma_load(&tl_ifmap, ga_input + tile.offset);
}

void TgZeroMaskKernel::store(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  ctx.tdma_store(&tl_ofmap, ga_output + tile.offset);
}

void TgZeroMaskKernel::compute_bf16(int32_t step_idx) {
  cvk_tiu_mul_param_t p1 = {0};
  p1.res_high = nullptr;
  p1.res_low = &tl_ofmap;
  p1.a = &tl_ifmap;
  p1.b_const.val = ctx.convert_fp32_to_bf16(1000000.0f);
  p1.b_const.is_signed = 1;
  p1.b_is_const = 1;
  p1.rshift_bits = 0;
  p1.layer_id = layer_id;
  p1.relu_enable = 0;
  ctx.tiu_mul(&p1);

  cvk_tiu_add_param_t p2 = {0};
  p2.res_high = nullptr;
  p2.res_low = &tl_ofmap;
  p2.a_high = nullptr;
  p2.a_low = &tl_ofmap;
  p2.b_is_const = true;
  p2.b_const.val = ctx.convert_fp32_to_bf16(1.0f);
  p2.rshift_bits = 0;
  p2.layer_id = layer_id;
  p2.relu_enable = 1;
  ctx.tiu_add(&p2);

  cvk_tiu_mul_param_t p3 = {0};
  p3.res_high = nullptr;
  p3.res_low = &tl_ofmap;
  p3.a = &tl_ofmap;
  p3.b_const.val = ctx.convert_fp32_to_bf16(-1.0f);
  p3.b_const.is_signed = 1;
  p3.b_is_const = 1;
  p3.rshift_bits = 0;
  p3.layer_id = layer_id;
  p3.relu_enable = 0;
  ctx.tiu_mul(&p3);

  cvk_tiu_add_param_t p4 = {0};
  p4.res_high = nullptr;
  p4.res_low = &tl_ofmap;
  p4.a_high = nullptr;
  p4.a_low = &tl_ofmap;
  p4.b_is_const = true;
  p4.b_const.val = ctx.convert_fp32_to_bf16(2.0f);
  p4.rshift_bits = 0;
  p4.layer_id = layer_id;
  p4.relu_enable = 1;
  ctx.tiu_add(&p4);

  if (positive) {
    cvk_tiu_mul_param_t p5 = {0};
    p5.res_high = nullptr;
    p5.res_low = &tl_ofmap;
    p5.a = &tl_ofmap;
    p5.b_const.val = ctx.convert_fp32_to_bf16(-1.0f);
    p5.b_const.is_signed = 1;
    p5.b_is_const = 1;
    p5.rshift_bits = 0;
    p5.layer_id = layer_id;
    p5.relu_enable = 0;
    ctx.tiu_mul(&p5);

    cvk_tiu_add_param_t p6 = {0};
    p6.res_high = nullptr;
    p6.res_low = &tl_ofmap;
    p6.a_high = nullptr;
    p6.a_low = &tl_ofmap;
    p6.b_is_const = true;
    p6.b_const.val = ctx.convert_fp32_to_bf16(1.0f);
    p6.rshift_bits = 0;
    p6.layer_id = layer_id;
    p6.relu_enable = 1;
    ctx.tiu_add(&p6);
  }
}

void TgZeroMaskKernel::compute_int8(int32_t step_idx) {
  ctx.tiu_zeros(layer_id, &tl_ofmap);
  cvk_tiu_add_param_t p1 = {0};
  p1.res_high = &tl_ofmap;
  p1.res_low = &tl_ifmap;
  p1.a_high = &tl_ofmap;
  p1.a_low = &tl_ifmap;
  p1.b_is_const = true;
  p1.b_const.val = 1;
  p1.rshift_bits = 0;
  p1.layer_id = layer_id;
  p1.relu_enable = 0;
  ctx.tiu_add(&p1);

  cvk_tiu_max_param_t param_relu = {0};
  param_relu.max = &tl_ifmap;
  param_relu.a = &tl_ifmap;
  param_relu.b_is_const = true;
  param_relu.b_const.val = (0);
  param_relu.b_const.is_signed = 1;
  param_relu.layer_id = layer_id;
  ctx.tiu_max(&param_relu);

  cvk_tiu_mul_param_t p3 = {0};
  p3.res_high = &tl_ofmap;
  p3.res_low = &tl_ifmap;
  p3.a = &tl_ifmap;
  p3.b_const.val = -1;
  p3.b_const.is_signed = 1;
  p3.b_is_const = 1;
  p3.rshift_bits = 0;
  p3.layer_id = layer_id;
  p3.relu_enable = 0;
  ctx.tiu_mul(&p3);

  cvk_tiu_add_param_t p2 = {0};
  p2.res_high = &tl_ofmap;
  p2.res_low = &tl_ifmap;
  p2.a_high = &tl_ofmap;
  p2.a_low = &tl_ifmap;
  p2.b_is_const = true;
  p2.b_const.val = 2;
  p2.rshift_bits = 0;
  p2.layer_id = layer_id;
  p2.relu_enable = 0;
  ctx.tiu_add(&p2);

  ctx.tiu_max(&param_relu);

  if (positive) {
    cvk_tiu_mul_param_t p4 = {0};
    p4.res_high = &tl_ofmap;
    p4.res_low = &tl_ifmap;
    p4.a = &tl_ifmap;
    p4.b_const.val = -1;
    p4.b_const.is_signed = 1;
    p4.b_is_const = 1;
    p4.rshift_bits = 0;
    p4.layer_id = layer_id;
    p4.relu_enable = 0;
    ctx.tiu_mul(&p4);

    cvk_tiu_add_param_t p5 = {0};
    p5.res_high = &tl_ofmap;
    p5.res_low = &tl_ifmap;
    p5.a_high = &tl_ofmap;
    p5.a_low = &tl_ifmap;
    p5.b_is_const = true;
    p5.b_const.val = 2;
    p5.rshift_bits = 0;
    p5.layer_id = layer_id;
    p5.relu_enable = 0;
    ctx.tiu_add(&p5);
    ctx.tiu_max(&param_relu);
  }
  cvk_tiu_mul_param_t p6 = {0};
  p6.res_high = &tl_ofmap;
  p6.res_low = &tl_ifmap;
  p6.a = &tl_ifmap;
  p6.b_const.val = 127;
  p6.b_const.is_signed = 1;
  p6.b_is_const = 1;
  p6.rshift_bits = 0;
  p6.layer_id = layer_id;
  p6.relu_enable = 0;
  ctx.tiu_mul(&p6);
  cvk_tiu_copy_param_t p = {0};
  p.src = &tl_ifmap;
  p.dst = &tl_ofmap;
  p.layer_id = layer_id;
  ctx.tiu_copy(&p);
}

void TgZeroMaskKernel::compute(int32_t step_idx) {
  if (fmt == CVK_FMT_BF16) {
    compute_bf16(step_idx);
  } else {
    compute_int8(step_idx);
  }
}

void TgZeroMaskKernel::schedule() {
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

void cvi_backend_zero_mask_kernel(const CviBackendContext &ctx,
                                  uint32_t layer_id, gaddr_t ga_input,
                                  gaddr_t ga_output, int n, int c, int h, int w,
                                  bool positive, cvk_fmt_t fmt) {
  TgZeroMaskKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_output, n, c, h, w, positive, fmt);
  kernel.selectTilePolicy();
  kernel.schedule();
}
