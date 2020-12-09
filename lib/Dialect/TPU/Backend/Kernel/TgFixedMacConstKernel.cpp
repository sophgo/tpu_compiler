/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgFixedMacConstKernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#define DEBUG_TYPE "bm1880v2_kernel_mac_const"
#define DEBUG_SPLIT "bm1880v2_kernel_mac_const_split"

// y = x * multiplier + const_val
void cvi_backend_tg_fixed_mac_const_kernel(const CviBackendContext &ctx,
                                           uint32_t layer_id, gaddr_t ga_input,
                                           gaddr_t ga_output, int n, int c,
                                           int h, int w, int multiplier,
                                           int const_val, bool do_relu) {

  int blob_num = 3; // 3 means we allocate input/output/output_high

  std::vector<CviBackendContext::tiling_info_t> tiles;
  ctx.tiling_packing(tiles, n, c, h, w, CVK_FMT_I8, blob_num, 0,
                     CviBackendContext::TilingAll);

  for (auto &tile : tiles) {
    cvk_tl_shape_t tl_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
    cvk_tl_t *tl_input = ctx.lmem_alloc_tensor(tl_shape, CVK_FMT_I8, 1);
    cvk_tl_t *tl_output = ctx.lmem_alloc_tensor(tl_shape, CVK_FMT_I8, 1);
    cvk_tl_t *tl_output_h = ctx.lmem_alloc_tensor(tl_shape, CVK_FMT_I8, 1);

    ctx.tdma_load(tl_input, ga_input + tile.offset);
    // clear high 8 bit
    cvk_tdma_g2l_tensor_fill_constant_param_t param = {0};
    param.dst = tl_output;
    param.layer_id = layer_id;
    param.constant = (uint16_t)const_val;
    ctx.tdma_g2l_tensor_fill_constant(&param);

    cvk_tdma_g2l_tensor_fill_constant_param_t param2 = {0};
    param2.dst = tl_output_h;
    param2.layer_id = layer_id;
    param2.constant = (uint16_t)0;
    ctx.tdma_g2l_tensor_fill_constant(&param2);

    cvk_tiu_mac_param_t p = {0};
    p.res_high = tl_output_h;
    p.res_low = tl_output;
    p.a = tl_input;
    p.res_is_int8 = 1;
    p.b_const.val = (int16_t)multiplier;
    p.b_is_const = 1;
    p.b_const.is_signed = 1;
    p.lshift_bits = 0;
    p.rshift_bits = 0;
    p.layer_id = layer_id;
    p.relu_enable = do_relu ? 1 : 0;
    ctx.tiu_mac(&p);

    ctx.tdma_store(tl_output, ga_output + tile.offset);
    ctx.lmem_free_tensor(tl_output_h);
    ctx.lmem_free_tensor(tl_output);
    ctx.lmem_free_tensor(tl_input);
  }
  return;
}

void cvi_backend_tg_bf16_mac_const_kernel(const CviBackendContext &ctx,
                                          uint32_t layer_id, gaddr_t ga_input,
                                          gaddr_t ga_output, int n, int c,
                                          int h, int w, float multiplier,
                                          float const_val, bool do_relu) {

  int blob_num = 2; // 2 means we allocate input/output
  std::vector<CviBackendContext::tiling_info_t> tiles;
  ctx.tiling_packing(tiles, n, c, h, w, CVK_FMT_BF16, blob_num, 0,
                     CviBackendContext::TilingAll);

  for (auto &tile : tiles) {

    cvk_tl_shape_t tl_shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
    cvk_tl_t *tl_input = ctx.lmem_alloc_tensor(tl_shape, CVK_FMT_BF16, 1);
    cvk_tl_t *tl_output = ctx.lmem_alloc_tensor(tl_shape, CVK_FMT_BF16, 1);

    ctx.tdma_load(tl_input, ga_input + tile.offset);

    cvk_tdma_g2l_tensor_fill_constant_param_t param = {0};
    param.dst = tl_output;
    param.layer_id = layer_id;
    param.constant = ctx.convert_fp32_to_bf16(const_val);
    ctx.tdma_g2l_tensor_fill_constant(&param);

    cvk_tiu_mac_param_t p = {0};
    p.res_high = nullptr;
    p.res_low = tl_output;
    p.a = tl_input;
    p.res_is_int8 = 0;
    p.b_const.val = ctx.convert_fp32_to_bf16(multiplier);
    p.b_is_const = 1;
    p.b_const.is_signed = 1;
    p.lshift_bits = 0;
    p.rshift_bits = 0;
    p.layer_id = layer_id;
    p.relu_enable = do_relu ? 1 : 0;
    ctx.tiu_mac(&p);

    ctx.tdma_store(tl_output, ga_output + tile.offset);
    ctx.lmem_free_tensor(tl_output);
    ctx.lmem_free_tensor(tl_input);
  }
  return;
}