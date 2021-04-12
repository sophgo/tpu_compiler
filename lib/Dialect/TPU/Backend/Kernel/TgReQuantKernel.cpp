/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgReQuantKernel.cpp
 * Description:
 */

#include "TgReQuantKernel.hpp"

#define DEBUG_TYPE "requant_kernel"


void TgReQuantKernel::init(uint32_t layer_id, gaddr_t ga_input,
            gaddr_t ga_output, int32_t n, int32_t c, int32_t h, int32_t w,
            int input_offset, int output_offset, float scale) {

  this->layer_id = layer_id;
  this->n = n;
  this->c = c;
  this->h = h;
  this->w = w;
  this->ga_input = ga_input;
  this->ga_output = ga_output;
  this->input_offset = input_offset;
  this->output_offset = output_offset;
  this->scale = scale;
  ctx.set_layer_id(layer_id);
}

void TgReQuantKernel::selectTilePolicy() {
  // 2 to do flip, one is input, one is output,
  int blob_num = 2 * (2);
  ctx.tiling_packing(tiles, n, c, h, w, CVK_FMT_BF16, blob_num, 0,
                     CviBackendContext::TilingAll);
}

cvk_tl_t *TgReQuantKernel::alloc_lmem(const cvk_tl_shape_t &shape,
                                      bool clean) const {
  cvk_tl_t *tl_mem = ctx.lmem_alloc_tensor(shape, CVK_FMT_BF16, 1);
  if (clean) {
    ctx.tiu_zeros(layer_id, tl_mem);
  }
  return tl_mem;
}

void TgReQuantKernel::allocLmem() {
  cvk_tl_shape_t input_shape = ctx.tl_shape_t4(
      tiles[0].n, tiles[0].c, tiles[0].h, tiles[0].w);
  tl_output[0] = alloc_lmem(input_shape, false);
  tl_output[1] = alloc_lmem(input_shape, false);
  tl_input[0] = alloc_lmem(input_shape, false);
  tl_input[1] = alloc_lmem(input_shape, false);
}
void TgReQuantKernel::deallocLmem() {
  ctx.lmem_free_tensor(tl_input[1]);
  ctx.lmem_free_tensor(tl_input[0]);
  ctx.lmem_free_tensor(tl_output[1]);
  ctx.lmem_free_tensor(tl_output[0]);
}

void TgReQuantKernel::compute(int32_t step_idx, int32_t flip) {
  auto &tile = tiles[step_idx];
  cvk_tl_t tl_ifmap = *tl_input[flip];
  tl_ifmap.shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  tl_ifmap.stride = ctx.tl_default_stride(tl_ifmap.shape, CVK_FMT_BF16, 1);

  cvk_tl_t tl_ofmap = *tl_output[flip];
  tl_ofmap.shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  tl_ofmap.stride = ctx.tl_default_stride(tl_ofmap.shape, CVK_FMT_BF16, 1);

  cvk_tiu_add_param_t p_input_offset = {0};
  p_input_offset.res_high = NULL;
  p_input_offset.res_low = &tl_ifmap;
  p_input_offset.a_high = NULL;
  p_input_offset.a_low = &tl_ifmap;
  p_input_offset.b_is_const = 1;
  p_input_offset.b_const.val = ctx.convert_fp32_to_bf16(input_offset);
  p_input_offset.relu_enable = 0;
  p_input_offset.layer_id = layer_id;
  ctx.tiu_add(&p_input_offset);

  cvk_tiu_mul_param_t p = {0};
  p.res_high = NULL;
  p.res_low = &tl_ifmap;
  p.a = &tl_ifmap;
  p.b_is_const = 1;
  p.b_const.val = ctx.convert_fp32_to_bf16(scale);
  p.relu_enable = 0;
  p.layer_id = layer_id;
  ctx.tiu_mul(&p);

  cvk_tiu_add_param_t p_offset = {0};
  p_offset.res_high = NULL;
  p_offset.res_low = &tl_ofmap;
  p_offset.a_high = NULL;
  p_offset.a_low = &tl_ifmap;
  p_offset.b_is_const = 1;
  p_offset.b_const.val = ctx.convert_fp32_to_bf16(output_offset);
  p_offset.relu_enable = 0;
  p_offset.layer_id = layer_id;
  ctx.tiu_add(&p_offset);
}

void TgReQuantKernel::load(int32_t step_idx, int32_t flip) {
  auto &tile = tiles[step_idx];
  cvk_tg_t src;
  src.start_address = ga_input + tile.offset * 0.5;
  src.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(src.start_address);
  src.int8_rnd_mode = 0;
  src.fmt = CVK_FMT_I8;
  src.shape = ctx.tg_shape_t4(tile.n, tile.c, tile.h, tile.w);
  src.stride = ctx.tg_default_stride(src.shape, src.fmt);
  cvk_tl_t tl_ifmap = *tl_input[flip];
  tl_ifmap.shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  tl_ifmap.stride = ctx.tl_default_stride(tl_ifmap.shape, CVK_FMT_BF16, 1);
  cvk_tdma_g2l_tensor_copy_param_t p1 = {0};
  p1.src = &src;
  p1.dst = &tl_ifmap;
  ctx.tdma_g2l_tensor_copy(&p1);
}

void TgReQuantKernel::store(int32_t step_idx, int32_t flip) {
  auto &tile = tiles[step_idx];
  cvk_tl_t tl_ofmap = *tl_output[flip];
  tl_ofmap.shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  tl_ofmap.stride = ctx.tl_default_stride(tl_ofmap.shape, tl_ofmap.fmt, 1);

  cvk_tg_t dst;
  dst.start_address = ga_output + tile.offset * 1 / 2;
  dst.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(dst.start_address);
  dst.fmt = CVK_FMT_I8;
  dst.shape = ctx.tg_shape_t4(tile.n, tile.c, tile.h, tile.w);
  dst.stride = ctx.tg_default_stride(dst.shape, dst.fmt);

  cvk_tdma_l2g_tensor_copy_param_t p1 = {0};
  p1.src = &tl_ofmap;
  p1.dst = &dst;
  ctx.tdma_l2g_tensor_copy(&p1);
}

void TgReQuantKernel::schedule() {
  allocLmem();
  int32_t total_steps = tiles.size();
  for (int32_t i = 0; i < total_steps + 2; i++) {
    ctx.parallel_enable();

    if (i - 1 >= 0 && i - 1 < total_steps) {
      compute(i - 1, 1 - flip);
    }
    if (i < total_steps) {
      load(i, flip);
    }
    if (i - 2 >= 0) {
      store(i - 2, flip);
    }
    flip = 1 - flip;
    ctx.parallel_disable();
  }
  deallocLmem();
}

void cvi_backend_tg_requant_kernel(const CviBackendContext &ctx,
                                   uint32_t layer_id, gaddr_t bottom_gaddr,
                                   gaddr_t top_gaddr, int input_n, int input_c,
                                   int input_h, int input_w, int input_offset,
                                   int output_offset, float scale) {
  TgReQuantKernel kernel(ctx);
  kernel.init(layer_id, bottom_gaddr, top_gaddr, input_n, input_c,
              input_h, input_w, input_offset, output_offset, scale);

  kernel.selectTilePolicy();
  kernel.schedule();
}
