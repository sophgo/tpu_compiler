/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: mac_const_kernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#define DEBUG_TYPE "bmnet_bm1880v2_bmkernel_mac_const"
#define DEBUG_SPLIT "bmnet_bm1880v2_bmkernel_mac_const_split"

// y = x * multiplier + const_val
void cvi_backend_tg_fixed_mac_const_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id,
    uint32_t layer_id, const uint32_t *depends, uint32_t depends_len,
    gaddr_t input_gaddr, gaddr_t output_gaddr, int input_n, int input_c,
    int input_h, int input_w, int multiplier, int const_val, bool do_relu) {

  int require_shape = input_n * input_c * input_h * input_w;
  int blob_num = 2; // 2 means we allocate input/output
  int coeff_lane_shape = 0;

  std::vector<std::pair<cvk_tl_shape_t, gaddr_t> > tiling_info;
  tiling_packing(ctx, require_shape, coeff_lane_shape, blob_num, CVK_FMT_I8, &tiling_info);

  for (size_t i = 0; i < tiling_info.size(); i++) {
    int n = tiling_info[i].first.n;
    int c = tiling_info[i].first.c;
    int h = tiling_info[i].first.h;
    int w = tiling_info[i].first.w;

    gaddr_t gaddr_offset = tiling_info[i].second;

    cvk_tl_shape_t tl_shape = ctx.shape_t4(n, c, h, w);
    cvk_tl_t *tl_output = ctx.lmem_alloc_tensor(tl_shape, CVK_FMT_I8, /*eu_align=*/1);
    cvk_tl_t *tl_input = ctx.lmem_alloc_tensor(tl_shape, CVK_FMT_I8, /*eu_align=*/1);

    ctx.tdma_load(tl_input, input_gaddr + gaddr_offset);
    // clear input_high 8 bit
    cvk_tdma_g2l_tensor_fill_constant_param_t param = {0};
    param.dst = tl_output;
    param.layer_id = layer_id;
    param.constant = (uint16_t)const_val;
    ctx.tdma_tg2l_tensor_fill_constant(&param);

    cvk_tiu_mac_param_t p = {0};
    p.res_high = tl_input;
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
    ctx.lmem_free_tensor(tl_input);
    ctx.lmem_free_tensor(tl_output);

  }
  return;
}
