/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgFixedLrnKernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include "backend/backend_tg_api.h"
#include <cmath>
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#define DEBUG_TYPE "cvi_backend_tg_bf16_lrn"

void cvi_backend_tg_bf16_lrn_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t input_gaddr,
    gaddr_t output_gaddr, gaddr_t exp_table_gaddr, gaddr_t mantissa_table_gaddr,
    int input_n, int input_c, int input_h, int input_w, int local_size,
    float alpha, float k) {

  llvm::errs() << llvm::format(
            "cvi_backend_tg_bf16_lrn:\n"
            "    input_gaddr 0x%lx, output_gaddr 0x%lx \n"
            "    exp_table 0x%lx, mantissa_table 0x%lx \n"
            "    in(%d, %d, %d, %d), size %d\n"
            "    alpha %f k %f\n",
            input_gaddr, output_gaddr, exp_table_gaddr, mantissa_table_gaddr,
            input_n, input_c, input_h, input_w, local_size, alpha, k);

  int blob_num = 4;
  cvk_fmt_t fmt = CVK_FMT_BF16;
  cvk_tl_shape_t table_shape = ctx.lut_table_shape(fmt);
  cvk_tg_shape_t gshape = ctx.tg_shape_t4(input_n, input_c, input_h, input_w);
  cvk_tg_stride_t gstride = ctx.tg_default_stride(gshape, fmt);

  cvk_tl_t *exp_table = ctx.lmem_alloc_tensor(table_shape, fmt, 1);
  cvk_tl_t *mantissa_table = ctx.lmem_alloc_tensor(table_shape, fmt, 1);
  ctx.tdma_load(exp_table, exp_table_gaddr);
  ctx.tdma_load(mantissa_table, mantissa_table_gaddr);

  uint32_t lmem_used = 2 * ctx.lmem_tensor_to_size(table_shape, fmt, 1);
  std::vector<CviBackendContext::tiling_info_t> tiles;
  ctx.tiling_packing(tiles, gshape, fmt, blob_num, lmem_used, CviBackendContext::TilingNHW);

  int move_counts = (local_size - 1) / 2;
  assert(move_counts <= input_c);

  for (auto &tile : tiles) {
    cvk_tl_shape_t lshape = ctx.tl_shape_t4(tile.n,tile.c,tile.h,tile.w);
    assert(lshape.c == (uint32_t)input_c);

    cvk_tl_t *bottom = ctx.lmem_alloc_tensor(lshape, fmt, 1);
    assert(bottom);
    cvk_tl_t *sum = ctx.lmem_alloc_tensor(lshape, fmt, 1);
    assert(sum);
    cvk_tl_t *shift_sum = ctx.lmem_alloc_tensor(lshape, fmt, 1);
    assert(shift_sum);
    cvk_tl_t *top = ctx.lmem_alloc_tensor(lshape, fmt, 1);
    assert(top);
    cvk_tl_t *tmp = top;

    uint64_t slice_bottom_gaddr = input_gaddr + tile.offset;
    uint64_t slice_top_gaddr = output_gaddr + tile.offset;

    ctx.tdma_load_stride(bottom, slice_bottom_gaddr, gstride);

    // y = x * ( k + sum(ax^2))^(-beta)
    // sum = x^2
    cvk_tiu_mul_param_t p0 = {0};
    p0.res_high = nullptr;
    p0.res_low = sum;
    p0.a = bottom;
    p0.b = bottom;
    p0.b_is_const = 0;
    p0.rshift_bits = 0;
    p0.layer_id = layer_id;
    p0.relu_enable = 0;
    ctx.tiu_mul(&p0);

    // sum = a(x^2)
    cvk_tiu_mul_param_t p1 = {0};
    p1.res_high = nullptr;
    p1.res_low =sum;
    p1.a = sum;
    p1.b_const.val = ctx.convert_fp32_to_bf16(alpha/local_size);
    p1.b_const.is_signed = 1;
    p1.b_is_const = 1;
    p1.rshift_bits = 0;
    p1.layer_id = layer_id;
    p1.relu_enable = 0;
    ctx.tiu_mul(&p1);

    // tmp = sum * 1.0
    cvk_tiu_mul_param_t p2 = {0};
    p2.res_high = nullptr;
    p2.res_low = tmp;
    p2.a = sum;
    p2.b_const.val = ctx.convert_fp32_to_bf16(1.0);
    p2.b_const.is_signed = 1;
    p2.b_is_const = 1;
    p2.rshift_bits = 0;
    p2.layer_id = layer_id;
    p2.relu_enable = 0;
    ctx.tiu_mul(&p2);

    // tmp = sum(ax^2)
    for (int step = 1; step <= move_counts && step < input_c; step++) {
        // sum shift c left -> shift_sum
        cvk_tdma_l2l_tensor_lrn_shift_param_t lrn_shift_p = {0};
        lrn_shift_p.dst = shift_sum;
        lrn_shift_p.src = tmp;
        lrn_shift_p.right_shift = false;
        lrn_shift_p.lrn_step = step;
        ctx.tdma_l2l_tensor_lrn_shift(&lrn_shift_p);

        // sum = shift_sum + sum
        cvk_tiu_mac_param_t p3 = {0};
        p3.res_high = nullptr;
        p3.res_low = sum;
        p3.res_is_int8 = 0;
        p3.a = shift_sum;
        p3.b_const.val = ctx.convert_fp32_to_bf16(1.0);
        p3.b_is_const = 1;
        p3.b_const.is_signed = 1;
        p3.lshift_bits = 0;
        p3.rshift_bits = 0;
        p3.layer_id = layer_id;
        p3.relu_enable = 0;
        ctx.tiu_mac(&p3);

        // sum shift c right -> shift_sum
        lrn_shift_p.dst = shift_sum;
        lrn_shift_p.src = tmp;
        lrn_shift_p.right_shift = true;
        lrn_shift_p.lrn_step = step;
        ctx.tdma_l2l_tensor_lrn_shift(&lrn_shift_p);

        // sum = shift_sum + sum
        cvk_tiu_mac_param_t p4 = {0};
        p4.res_high = nullptr;
        p4.res_low = sum;
        p4.res_is_int8 = 0;
        p4.a = shift_sum;
        p4.b_const.val = ctx.convert_fp32_to_bf16(1.0);
        p4.b_is_const = 1;
        p4.b_const.is_signed = 1;
        p4.lshift_bits = 0;
        p4.rshift_bits = 0;
        p4.relu_enable = 0;
        ctx.tiu_mac(&p4);
    }

    // sum = (k + sum(ax^2))
    cvk_tiu_add_param_t p5 = {0};
    p5.res_high = nullptr;
    p5.res_low = sum;
    p5.a_high = nullptr;
    p5.a_low = sum;
    p5.b_is_const = true;
    p5.b_const.val = ctx.convert_fp32_to_bf16(k);
    p5.rshift_bits = 0;
    p5.layer_id = layer_id;
    p5.relu_enable = false;
    ctx.tiu_add(&p5);

    // (k+sum(ax^2))^(-beta) = tmp^(-beta)
    // we find the exp and mantissa for tmp, and mul them
    // ==> lut: exp * lut: mantissa
    cvk_tdma_l2l_tensor_copy_param_t p6;
    // move high 8 bits to low 8 bit so
    // that we can do table lookup for exp
    memset(&p6, 0x00, sizeof(cvk_tdma_l2l_tensor_copy_param_t));
    p6.dst = shift_sum;
    p6.src = sum;
    p6.mv_lut_base = false;
    p6.mv_lut_idx = true;
    p6.layer_id = layer_id;
    ctx.tdma_l2l_tensor_copy(&p6);

    // tmp = lut: exp
    cvk_tiu_lookup_table_param_t p7 = {0};
    p7.ofmap = tmp;
    p7.ifmap = shift_sum;
    p7.table = exp_table;
    p7.layer_id = layer_id;
    ctx.tiu_lookup_table(&p7);

    // shift_sum = lut: mantissa
    cvk_tiu_lookup_table_param_t p8 = {0};
    p8.ofmap = shift_sum;
    p8.ifmap = sum;
    p8.table = mantissa_table;
    p8.layer_id = layer_id;
    ctx.tiu_lookup_table(&p8);

    // (1+sum(ax^2))^(-beta) = tmp * shift_sum
    cvk_tiu_mul_param_t p9 = {0};
    p9.res_high = nullptr;
    p9.res_low = top;
    p9.a = tmp;
    p9.b = shift_sum;
    p9.b_is_const = 0;
    p9.rshift_bits = 0;
    p9.layer_id = layer_id;
    p9.relu_enable = 0;
    ctx.tiu_mul(&p9);

    // x * (1+sum(ax^2))^(-beta) = top * bottom
    cvk_tiu_mul_param_t p10 = {0};
    p10.res_high = nullptr;
    p10.res_low = top;
    p10.a = top;
    p10.b = bottom;
    p10.b_is_const = 0;
    p10.rshift_bits = 0;
    p10.layer_id = layer_id;
    p10.relu_enable = 0;
    ctx.tiu_mul(&p10);

    // Original global memory shape used to calculate global stride
    // Asign global memory shape as local memory's
    ctx.tdma_store_stride(top, slice_top_gaddr, gstride);

    ctx.lmem_free_tensor(top);
    ctx.lmem_free_tensor(shift_sum);
    ctx.lmem_free_tensor(sum);
    ctx.lmem_free_tensor(bottom);
  }
  ctx.lmem_free_tensor(mantissa_table);
  ctx.lmem_free_tensor(exp_table);
}
