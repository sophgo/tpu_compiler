/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: bf16_TgReluKernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>

#define DEBUG_TYPE "cvi_backend_leakyrelu_kernel"

void cvi_backend_tg_bf16_leakyrelu_kernel(const CviBackendContext &ctx, uint32_t layer_id,
                                      gaddr_t ga_bottom, gaddr_t ga_top, float ga_negative_slope,
                                      int input_n, int input_c, int input_h, int input_w) {
  LLVM_DEBUG(llvm::errs() << llvm::format(
      "cvi_backend_tg_bf16_leakyrelu_kernel:\n"
      "    layer_id %d\n"
      "    bottom = %lx, top = %lx, negative_slope = %f\n"
      "    nchw = (%d, %d, %d, %d)\n",
      layer_id, ga_bottom, ga_top, ga_negative_slope, input_n, input_c, input_h, input_w););

  cvk_fmt_t fmt = CVK_FMT_BF16;

  // 2x for input, output
  int blob_num = 2; // input, output
  int require_shape = input_n * input_c * input_h * input_w;
  int coeff_lane_shape = 0;
  cvk_tg_shape_t tg_shape = ctx.tg_shape_t4(input_n, input_c, input_h, input_w);

  std::vector<std::pair<cvk_tl_shape_t, gaddr_t>> tiles;
  ctx.tiling_packing(require_shape, coeff_lane_shape, blob_num, fmt,
                     &tiles, ctx.TilingDimAll, &tg_shape);

  cvk_tl_shape_t max_shape =
      ctx.tl_shape_t4(tiles[0].first.n, tiles[0].first.c,
                      tiles[0].first.h, tiles[0].first.w);
  cvk_tl_t *tl_input = ctx.lmem_alloc_tensor(max_shape, fmt, /*eu_align=*/1);
  cvk_tl_t *tl_output = ctx.lmem_alloc_tensor(max_shape, fmt, /*eu_align=*/1);

  for (auto &tile : tiles) {
    LLVM_DEBUG(llvm::errs()
        << "loop, tiled shape(" << tile.first.n
        << ", " << tile.first.c << ", " << tile.first.h
        << ", " << tile.first.w << "), offset " << tile.second << "\n");

    cvk_tl_shape_t tiled_shape = ctx.tl_shape_t4(tile.first.n, tile.first.c,
                                                 tile.first.h, tile.first.w);
    cvk_tl_t bottom;
    ctx.lmem_init_tensor(&bottom, tiled_shape, fmt, /*eu_align=*/1);
    bottom.start_address = tl_input->start_address;

    cvk_tl_t relu;
    ctx.lmem_init_tensor(&relu, tiled_shape, fmt, /*eu_align=*/1);
    relu.start_address = tl_output->start_address;

    uint64_t offset = tile.second;
    ctx.tdma_load(&bottom, ga_bottom + offset);

    // 0. relu = bottom * slope
    // 1. relu = max(bottom, relu)

    // 0. relu = bottom * slope
    cvk_tiu_mul_param_t p1 = {0};
    p1.res_high = nullptr; //useless
    p1.res_low = &relu;
    p1.a = &bottom;
    p1.b_const.val = ctx.convert_fp32_to_bf16(ga_negative_slope);
    p1.b_const.is_signed = true;
    p1.b_is_const = true;
    p1.rshift_bits = 0;
    p1.layer_id = layer_id;
    p1.relu_enable = 0;
    ctx.tiu_mul(&p1);

    // 1. relu = max(bottom, relu)
    if(ga_negative_slope <= 1) {
      cvk_tiu_max_param_t p13 = {0};
      p13.max = &relu;
      p13.a = &bottom;
      p13.b_is_const = 0;
      p13.b_const.is_signed = 1;
      p13.b = &relu;
      p13.layer_id = layer_id;
      ctx.tiu_max(&p13);
    } else {
      cvk_tiu_min_param_t p13 = {0};
      p13.min = &relu;
      p13.a = &bottom;
      p13.b_is_const = 0;
      p13.b_const.is_signed = 1;
      p13.b = &relu;
      p13.layer_id = layer_id;
      ctx.tiu_min(&p13);
    }

    // move result to global
    ctx.tdma_store(&relu, ga_top + offset);
  }

  ctx.lmem_free_tensor(tl_output);
  ctx.lmem_free_tensor(tl_input);
}
