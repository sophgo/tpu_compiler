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
void cvi_backend_tg_fixed_mac_const_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t input_gaddr,
    gaddr_t output_gaddr, int input_n, int input_c, int input_h, int input_w,
    int multiplier, int const_val, bool do_relu) {

  int require_shape = input_n * input_c * input_h * input_w;
  int blob_num = 3; // 3 means we allocate input/output/output_high
  int coeff_lane_shape = 0;

  std::vector<std::pair<cvk_tl_shape_t, gaddr_t> > tiling_info;
  ctx.tiling_packing(require_shape, coeff_lane_shape, blob_num, CVK_FMT_I8, &tiling_info);

  for (size_t i = 0; i < tiling_info.size(); i++) {
    int n = tiling_info[i].first.n;
    int c = tiling_info[i].first.c;
    int h = tiling_info[i].first.h;
    int w = tiling_info[i].first.w;

    gaddr_t gaddr_offset = tiling_info[i].second;

    cvk_tl_shape_t tl_shape = ctx.tl_shape_t4(n, c, h, w);
    cvk_tl_t *tl_input = ctx.lmem_alloc_tensor(tl_shape, CVK_FMT_I8, /*eu_align=*/1);
    cvk_tl_t *tl_output = ctx.lmem_alloc_tensor(tl_shape, CVK_FMT_I8, /*eu_align=*/1);
    cvk_tl_t *tl_output_h = ctx.lmem_alloc_tensor(tl_shape, CVK_FMT_I8, /*eu_align=*/1);


    ctx.tdma_load(tl_input, input_gaddr + gaddr_offset);
    // clear input_high 8 bit
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

    ctx.tdma_store(tl_output, output_gaddr + gaddr_offset);
    ctx.lmem_free_tensor(tl_output_h);
    ctx.lmem_free_tensor(tl_output);
    ctx.lmem_free_tensor(tl_input);

  }
  return;
}

void cvi_backend_tg_bf16_mac_const_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t input_gaddr,
    gaddr_t output_gaddr, int input_n, int input_c, int input_h, int input_w,
    float multiplier, float const_val, bool do_relu) {

  int require_shape = input_n * input_c * input_h * input_w;
  int blob_num = 2; // 2 means we allocate input/output
  int coeff_lane_shape = 0;

  std::vector<std::pair<cvk_tl_shape_t, gaddr_t> > tiling_info;
  ctx.tiling_packing(require_shape, coeff_lane_shape, blob_num, CVK_FMT_BF16, &tiling_info);

  for (size_t i = 0; i < tiling_info.size(); i++) {
    int n = tiling_info[i].first.n;
    int c = tiling_info[i].first.c;
    int h = tiling_info[i].first.h;
    int w = tiling_info[i].first.w;

    gaddr_t gaddr_offset = tiling_info[i].second;

    cvk_tl_shape_t tl_shape = ctx.tl_shape_t4(n, c, h, w);
    cvk_tl_t *tl_input = ctx.lmem_alloc_tensor(tl_shape, CVK_FMT_BF16, /*eu_align=*/1);
    cvk_tl_t *tl_output = ctx.lmem_alloc_tensor(tl_shape, CVK_FMT_BF16, /*eu_align=*/1);


    ctx.tdma_load(tl_input, input_gaddr + gaddr_offset);

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

    ctx.tdma_store(tl_output, output_gaddr + gaddr_offset);
    ctx.lmem_free_tensor(tl_output);
    ctx.lmem_free_tensor(tl_input);

  }
  return;
}