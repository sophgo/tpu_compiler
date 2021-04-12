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

void cvi_backend_tg_fixed_lrn_kernel(const CviBackendContext &ctx,
                                     uint32_t layer_id, gaddr_t ga_input,
                                     gaddr_t ga_output, gaddr_t sqr_lut_gaddr,
                                     gaddr_t power_lut_gaddr, int input_n,
                                     int input_c, int input_h, int input_w,
                                     int local_size, int sum_right_shift_width,
                                     int lrn_right_shift_width, int quant_data0,
                                     int quant_data1) {

  int blob_num = 6;
  cvk_tl_shape_t table_shape = ctx.lut_table_shape(CVK_FMT_I8);
  cvk_tl_t *sqr_lut_table = ctx.lmem_alloc_tensor(table_shape, CVK_FMT_I8, 1);
  cvk_tl_t *power_lut_table = ctx.lmem_alloc_tensor(table_shape, CVK_FMT_I8, 1);
  ctx.tdma_load(power_lut_table, power_lut_gaddr);
  ctx.tdma_load(sqr_lut_table, sqr_lut_gaddr);
  uint32_t lmem_used = 2 * ctx.lmem_tensor_to_size(table_shape, CVK_FMT_I8, 1);

  cvk_tg_shape_t gshape = ctx.tg_shape_t4(input_n, input_c, input_h, input_w);
  cvk_tg_stride_t gstride = ctx.tg_default_stride(gshape, CVK_FMT_I8);
  std::vector<CviBackendContext::tiling_info_t> tiles;
  ctx.tiling_packing(tiles, gshape, CVK_FMT_I8, blob_num, lmem_used,
                     CviBackendContext::TilingNHW);

  for (auto &tile : tiles) {
    cvk_tl_shape_t lshape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);

    cvk_tl_t *bottom = ctx.lmem_alloc_tensor(lshape, CVK_FMT_U8, 1);
    cvk_tl_t *sum = ctx.lmem_alloc_tensor(lshape, CVK_FMT_U8, 1);
    cvk_tl_t *sum_high = ctx.lmem_alloc_tensor(lshape, CVK_FMT_U8, 1);
    cvk_tl_t *top = ctx.lmem_alloc_tensor(lshape, CVK_FMT_U8, 1);
    cvk_tl_t *top_high = ctx.lmem_alloc_tensor(lshape, CVK_FMT_U8, 1);
    cvk_tl_t *top_high_high = ctx.lmem_alloc_tensor(lshape, CVK_FMT_U8, 1);

    ctx.tdma_load_stride(bottom, ga_input + tile.offset, gstride);

    // lut:x^2*alpha/local_size
    // move data from gmem to lmem
    cvk_tiu_lookup_table_param_t p1 = {0};
    p1.ofmap = top;
    p1.ifmap = bottom;
    p1.table = sqr_lut_table;
    p1.layer_id = layer_id;
    ctx.tiu_lookup_table(&p1);
    // copy top => sum
    cvk_tiu_copy_param_t p2 = {0};
    p2.dst = sum;
    p2.src = top;
    p2.layer_id = layer_id;
    ctx.tiu_copy(&p2);

    // clean sum_high and top_high_high
    ctx.tiu_zeros(layer_id, sum_high);
    ctx.tiu_zeros(layer_id, top_high_high);

    // use lrn shift to add different feature map data
    // sum(x^2*alpha/local_size),16bits
    int move_counts = (local_size - 1) / 2;
    assert(move_counts <= input_c);
    for (int step = 1; step <= move_counts && step < input_c; step++) {
      cvk_tdma_l2l_tensor_lrn_shift_param_t p4 = {0};
      p4.dst = top_high_high;
      p4.src = top;
      p4.right_shift = false;
      p4.lrn_step = step;
      ctx.tdma_l2l_tensor_lrn_shift(&p4);

      cvk_tiu_mac_param_t p5 = {0};
      p5.res_high = sum_high;
      p5.res_low = sum;
      p5.res_is_int8 = 0;
      p5.a = top_high_high;
      p5.b_const.val = 1;
      p5.b_is_const = 1;
      p5.b_const.is_signed = 0;
      p5.lshift_bits = 0;
      p5.rshift_bits = 0;
      p5.layer_id = layer_id;
      p5.relu_enable = 0;
      ctx.tiu_mac(&p5);

      cvk_tdma_l2l_tensor_lrn_shift_param_t p6 = {0};
      p6.dst = top_high;
      p6.src = top;
      p6.right_shift = true;
      p6.lrn_step = step;
      ctx.tdma_l2l_tensor_lrn_shift(&p6);

      cvk_tiu_mac_param_t p7 = {0};
      p7.res_high = sum_high;
      p7.res_low = sum;
      p7.res_is_int8 = 0;
      p7.a = top_high;
      p7.b_const.val = 1;
      p7.b_is_const = 1;
      p7.b_const.is_signed = 0;
      p7.lshift_bits = 0;
      p7.rshift_bits = 0;
      p7.layer_id = layer_id;
      p7.relu_enable = 0;
      ctx.tiu_mac(&p7);
    }
    // 16bits higher  8bits,
    cvk_tiu_mul_param_t p8 = {0};
    p8.res_high = top_high;
    p8.res_low = sum_high;
    p8.a = sum_high;
    p8.b_const.val = quant_data0;
    p8.b_const.is_signed = 0;
    p8.b_is_const = 1;
    p8.rshift_bits = 0;
    p8.layer_id = layer_id;
    p8.relu_enable = 0;
    ctx.tiu_mul(&p8);

    cvk_tiu_mac_param_t p9 = {0};
    p9.res_high = top_high;
    p9.res_low = sum_high;
    p9.res_is_int8 = true;
    p9.a = sum;
    p9.b_const.val = quant_data0;
    p9.b_is_const = 1;
    p9.b_const.is_signed = 0;
    p9.lshift_bits = 8;
    p9.rshift_bits = sum_right_shift_width;
    p9.layer_id = layer_id;
    p9.relu_enable = 0;
    ctx.tiu_mac(&p9);

    // scale=lut:(k+sum)^(-beta)
    cvk_tiu_lookup_table_param_t p10 = {0};
    p10.ofmap = top;
    p10.ifmap = sum_high;
    p10.table = power_lut_table;
    p10.layer_id = layer_id;
    ctx.tiu_lookup_table(&p10);

    // Y=x*scale*threshold_x_quantized[1]>>lrn_right_shift_width
    cvk_tiu_mul_param_t p11 = {0};
    p11.res_high = top_high;
    p11.res_low = top;
    p11.a = bottom;
    p11.b = top;
    p11.b_is_const = 0;
    p11.rshift_bits = 0;
    p11.layer_id = layer_id;
    p11.relu_enable = 0;
    ctx.tiu_mul(&p11);

    // 16bits higher  8bits,
    cvk_tiu_mul_param_t p12 = {0};
    p12.res_high = top_high_high;
    p12.res_low = top_high;
    p12.a = top_high;
    p12.b_const.val = quant_data1;
    p12.b_const.is_signed = 0;
    p12.b_is_const = 1;
    p12.rshift_bits = 0;
    p12.layer_id = layer_id;
    p12.relu_enable = 0;
    ctx.tiu_mul(&p12);

    cvk_tiu_mac_param_t p13 = {0};
    p13.res_high = top_high_high;
    p13.res_low = top_high;
    p13.res_is_int8 = true;
    p13.a = top;
    p13.b_const.val = quant_data1;
    p13.b_is_const = 1;
    p13.b_const.is_signed = 0;
    p13.lshift_bits = 8;
    p13.rshift_bits = lrn_right_shift_width;
    p13.layer_id = layer_id;
    p13.relu_enable = 0;
    ctx.tiu_mac(&p13);

    cvk_tiu_min_param_t p14 = {0};
    p14.min = top_high;
    p14.a = top_high;
    p14.b_is_const = 1;
    p14.b_const.val = 127;
    p14.b_const.is_signed = 0;
    p14.layer_id = layer_id;
    ctx.tiu_min(&p14);
    // Original global memory shape used to calculate global stride
    // Asign global memory shape as local memory's
    ctx.tdma_store_stride(top_high, ga_output + tile.offset, gstride);

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
