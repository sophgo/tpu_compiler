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

void cvi_backend_tg_fixed_lrn_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t input_gaddr,
    gaddr_t output_gaddr, gaddr_t sqr_lut_gaddr, gaddr_t power_lut_gaddr,
    int input_n, int input_c, int input_h, int input_w, int local_size,
    int sum_right_shift_width, int lrn_right_shift_width, int quant_data0,
    int quant_data1) {

  int blob_num = 6;
  int require_shape = 0;
  cvk_tl_shape_t table_shape = ctx.lut_table_shape(CVK_FMT_I8);
  int coeff_lane_shape = 2 * table_shape.h * table_shape.w; // for sqr_lut and power_lut
  cvk_tg_shape_t gshape = ctx.tg_shape_t4(input_n, input_c, input_h, input_w);
  cvk_tg_stride_t gstride = ctx.tg_default_stride(gshape, CVK_FMT_I8);
  std::vector<std::pair<cvk_tl_shape_t, gaddr_t>> tiling_info;
  ctx.tiling_packing(require_shape, coeff_lane_shape, blob_num, CVK_FMT_I8,
                     &tiling_info, ctx.TilingDimNH, &gshape);

  cvk_tl_t *sqr_lut_table = ctx.lmem_alloc_tensor(table_shape, CVK_FMT_I8, 1);
  cvk_tl_t *power_lut_table = ctx.lmem_alloc_tensor(table_shape, CVK_FMT_I8, 1);
  ctx.tdma_load(power_lut_table, power_lut_gaddr);
  ctx.tdma_load(sqr_lut_table, sqr_lut_gaddr);

  int move_counts = (local_size - 1) / 2;
  assert(move_counts <= input_c);
  for (size_t i = 0; i < tiling_info.size(); i++) {

    cvk_tl_shape_t lshape = tiling_info[i].first;
    assert(lshape.c == (uint32_t)input_c);

    cvk_tl_t *bottom = ctx.lmem_alloc_tensor(lshape, CVK_FMT_U8, 1);
    cvk_tl_t *sum = ctx.lmem_alloc_tensor(lshape, CVK_FMT_U8, 1);
    cvk_tl_t *sum_high = ctx.lmem_alloc_tensor(lshape, CVK_FMT_U8, 1);
    cvk_tl_t *top = ctx.lmem_alloc_tensor(lshape, CVK_FMT_U8, 1);
    cvk_tl_t *top_high = ctx.lmem_alloc_tensor(lshape, CVK_FMT_U8, 1);
    cvk_tl_t *top_high_high = ctx.lmem_alloc_tensor(lshape, CVK_FMT_U8, 1);

    uint64_t slice_bottom_gaddr = input_gaddr + tiling_info[i].second;
    uint64_t slice_top_gaddr = output_gaddr + tiling_info[i].second;

    ctx.tdma_load_stride(bottom, slice_bottom_gaddr, gstride);

    // lut:x^2*alpha/local_size
    // move data from gmem to lmem
    cvk_tiu_lookup_table_param_t p12 = {0};
    p12.ofmap = top;
    p12.ifmap = bottom;
    p12.table = sqr_lut_table;
    p12.layer_id = layer_id;
    ctx.tiu_lookup_table(&p12);

    cvk_tiu_copy_param_t p10 = {0};
    p10.dst = sum;
    p10.src = top;
    p10.layer_id = layer_id;
    ctx.tiu_copy(&p10);

    cvk_tiu_mul_param_t p = {0};
    p.res_high = top_high_high;
    p.res_low = sum_high;
    p.a = top;
    p.b_const.val = 0;
    p.b_const.is_signed = 0;
    p.b_is_const = 1;
    p.rshift_bits = 0;
    p.layer_id = layer_id;
    p.relu_enable = 0;
    ctx.tiu_mul(&p); // sum_high initialize 0

    // use lrn shift to add different feature map data
    // sum(x^2*alpha/local_size),16bits
    for (int step = 1; step <= move_counts && step < input_c; step++) {
      cvk_tdma_l2l_tensor_lrn_shift_param_t lrn_shift_p = {0};
      lrn_shift_p.dst = top_high_high;
      lrn_shift_p.src = top;
      lrn_shift_p.right_shift = false;
      lrn_shift_p.lrn_step = step;
      ctx.tdma_l2l_tensor_lrn_shift(&lrn_shift_p);

      cvk_tiu_mac_param_t p3 = {0};
      p3.res_high = sum_high;
      p3.res_low = sum;
      p3.res_is_int8 = 0;
      p3.a = top_high_high;
      p3.b_const.val = 1;
      p3.b_is_const = 1;
      p3.b_const.is_signed = 0;
      p3.lshift_bits = 0;
      p3.rshift_bits = 0;
      p3.layer_id = layer_id;
      p3.relu_enable = 0;
      ctx.tiu_mac(&p3);

      lrn_shift_p.dst = top_high;
      lrn_shift_p.src = top;
      lrn_shift_p.right_shift = true;
      lrn_shift_p.lrn_step = step;
      ctx.tdma_l2l_tensor_lrn_shift(&lrn_shift_p);

      p3.res_high = sum_high;
      p3.res_low = sum;
      p3.res_is_int8 = 0;
      p3.a = top_high;
      p3.b_const.val = 1;
      p3.b_is_const = 1;
      p3.b_const.is_signed = 0;
      p3.lshift_bits = 0;
      p3.rshift_bits = 0;
      p3.layer_id = layer_id;
      p3.relu_enable = 0;
      ctx.tiu_mac(&p3);
    }
    // 16bits higher  8bits,
    p.res_high = top_high;
    p.res_low = sum_high;
    p.a = sum_high;
    p.b_const.val = quant_data0;
    p.b_const.is_signed = 0;
    p.b_is_const = 1;
    p.rshift_bits = 0;
    p.layer_id = layer_id;
    p.relu_enable = 0;
    ctx.tiu_mul(&p);

    cvk_tiu_mac_param_t p3 = {0};
    p3.res_high = top_high;
    p3.res_low = sum_high;
    p3.res_is_int8 = true;
    p3.a = sum;
    p3.b_const.val = quant_data0;
    p3.b_is_const = 1;
    p3.b_const.is_signed = 0;
    p3.lshift_bits = 8;
    p3.rshift_bits = sum_right_shift_width;
    p3.layer_id = layer_id;
    p3.relu_enable = 0;
    ctx.tiu_mac(&p3);

    // scale=lut:(k+sum)^(-beta)
    p12.ofmap = top;
    p12.ifmap = sum_high;
    p12.table = power_lut_table;
    p12.layer_id = layer_id;
    ctx.tiu_lookup_table(&p12);

    // Y=x*scale*threshold_x_quantized[1]>>lrn_right_shift_width
    cvk_tiu_mul_param_t p1 = {0};
    p1.res_high = top_high;
    p1.res_low = top;
    p1.a = bottom;
    p1.b = top;
    p1.b_is_const = 0;
    p1.rshift_bits = 0;
    p1.layer_id = layer_id;
    p1.relu_enable = 0;
    ctx.tiu_mul(&p1);

    // 16bits higher  8bits,
    p.res_high = top_high_high;
    p.res_low = top_high;
    p.a = top_high;
    p.b_const.val = quant_data1;
    p.b_const.is_signed = 0;
    p.b_is_const = 1;
    p.rshift_bits = 0;
    p.layer_id = layer_id;
    p.relu_enable = 0;
    ctx.tiu_mul(&p);

    p3.res_high = top_high_high;
    p3.res_low = top_high;
    p3.res_is_int8 = true;
    p3.a = top;
    p3.b_const.val = quant_data1;
    p3.b_is_const = 1;
    p3.b_const.is_signed = 0;
    p3.lshift_bits = 8;
    p3.rshift_bits = lrn_right_shift_width;
    p3.layer_id = layer_id;
    p3.relu_enable = 0;
    ctx.tiu_mac(&p3);

    cvk_tiu_min_param_t p7 = {0};
    p7.min = top_high;
    p7.a = top_high;
    p7.b_is_const = 1;
    p7.b_const.val = 127;
    p7.b_const.is_signed = 0;
    p7.layer_id = layer_id;
    ctx.tiu_min(&p7);

    // Original global memory shape used to calculate global stride
    // Asign global memory shape as local memory's
    ctx.tdma_store_stride(top_high, slice_top_gaddr, gstride);

    ctx.lmem_free_tensor(top_high_high);
    ctx.lmem_free_tensor(top_high);
    ctx.lmem_free_tensor(top);
    ctx.lmem_free_tensor(sum_high);
    ctx.lmem_free_tensor(sum);
    ctx.lmem_free_tensor(bottom);
  }
  ctx.lmem_free_tensor(power_lut_table);
  ctx.lmem_free_tensor(sqr_lut_table);
}
