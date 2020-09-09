/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: tl_lrn.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include "backend/backend_tl_api.h"

#define DEBUG_TYPE "tl_lrn"

void cvi_backend_tl_lrn(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t ifmap_laddr, laddr_t ofmap_laddr, laddr_t sqr_lut_laddr,
    laddr_t power_lut_laddr, laddr_t working_laddr,
    int input_n, int input_c, int input_h, int input_w, int size,
    int8_t sum_rshift_i8, int8_t lrn_rshift_i8,
    int8_t *m_i8) {
  LLVM_DEBUG(llvm::errs() << llvm::format(
            "cvi_backend_tl_lrn:\n"
            "    ifmap_laddr 0x%lx, ofmap_laddr 0x%lx"
            "    power_lut_laddr 0x%lx, power_lut_laddr 0x%lx, working_laddr 0x%lx"
            "    in(%d, %d, %d, %d), size %d\n"
            "    sum_right_shit_width %d, lrn_rshift_i8 %d\n",
            ifmap_laddr, ofmap_laddr, sqr_lut_laddr, power_lut_laddr,
            working_laddr, input_n,input_c, input_h, input_w, size,
            sum_rshift_i8, lrn_rshift_i8));

  int move_counts = (size - 1) / 2;
  cvk_tl_shape_t lshape =
          {static_cast<uint32_t>(input_n), static_cast<uint32_t>(input_c),
           static_cast<uint32_t>(input_h), static_cast<uint32_t>(input_w)};

  // cvk_tl_shape_t table_shape = {1, 32, 16, 16};
  cvk_tl_shape_t table_shape =
          {1, static_cast<uint32_t>(NPU_NUM), 16, 16};

  cvk_tl_t bottom;
  bottom.start_address = ifmap_laddr;
  bottom.fmt = CVK_FMT_U8;
  bottom.shape = lshape;
  bottom.stride = ctx.tl_default_stride(lshape, CVK_FMT_I8, 1);

  cvk_tl_t top;
  top.start_address = ofmap_laddr;
  top.fmt = CVK_FMT_U8;
  top.shape = lshape;
  top.stride = ctx.tl_default_stride(lshape, CVK_FMT_I8, 1);

  cvk_tl_t sqr_lut;
  sqr_lut.start_address = sqr_lut_laddr;
  sqr_lut.fmt = CVK_FMT_I8;
  sqr_lut.shape = table_shape;
  sqr_lut.stride = ctx.tl_default_stride(table_shape, CVK_FMT_I8, 1);

  cvk_tl_t pwr_lut;
  pwr_lut.start_address = power_lut_laddr;
  pwr_lut.fmt = CVK_FMT_I8;
  pwr_lut.shape = table_shape;
  pwr_lut.stride = ctx.tl_default_stride(table_shape, CVK_FMT_I8, 1);

  cvk_tl_t sum;
  sum.start_address = working_laddr;
  sum.fmt = CVK_FMT_U8;
  sum.shape = lshape;
  sum.stride = ctx.tl_default_stride(lshape, CVK_FMT_I8, 1);
  uint32_t sum_size = __get_lmem_usage(ctx, lshape.n, lshape.c, lshape.h, lshape.w);

  cvk_tl_t sum_high;
  sum_high.start_address = sum.start_address + sum_size;  // after sum
  sum_high.fmt = CVK_FMT_U8;
  sum_high.shape = lshape;
  sum_high.stride = ctx.tl_default_stride(lshape, CVK_FMT_I8, 1);
  uint32_t sum_high_size = __get_lmem_usage(ctx, lshape.n, lshape.c, lshape.h, lshape.w);

  cvk_tl_t shift_sum;
  shift_sum.start_address = sum_high.start_address + sum_high_size;  // after sum_high
  shift_sum.fmt = CVK_FMT_U8;
  shift_sum.shape = lshape;
  shift_sum.stride = ctx.tl_default_stride(lshape, CVK_FMT_I8, 1);
  uint32_t shift_sum_size = __get_lmem_usage(ctx, lshape.n, lshape.c, lshape.h, lshape.w);

  cvk_tl_t top_high;
  top_high.start_address = shift_sum.start_address + shift_sum_size;  // after shift_sum
  top_high.fmt = CVK_FMT_U8;
  top_high.shape = lshape;
  top_high.stride = ctx.tl_default_stride(lshape, CVK_FMT_I8, 1);
  uint32_t top_high_size = __get_lmem_usage(ctx, lshape.n, lshape.c, lshape.h, lshape.w);

  cvk_tl_t top_high_high;
  top_high_high.start_address = top_high.start_address + top_high_size;  // after top_high
  top_high_high.fmt = CVK_FMT_U8;
  top_high_high.shape = lshape;
  top_high_high.stride = ctx.tl_default_stride(lshape, CVK_FMT_I8, 1);
  uint32_t top_high_high_size = __get_lmem_usage(ctx, lshape.n, lshape.c, lshape.h, lshape.w);

  // Should no exceed local memory size
  assert((top_high_high.start_address + top_high_high_size) <= (uint32_t)(LOCAL_MEM_SIZE));

  cvk_tiu_lookup_table_param_t p12 = {0};
  p12.ofmap = &top;
  p12.ifmap = &bottom;
  p12.table = &sqr_lut;
  p12.layer_id = layer_id;
  ctx.tiu_lookup_table(&p12);

  cvk_tiu_mul_param_t p = {0};
  p.res_high = nullptr;
  p.res_low = &sum;
  p.a = &top;
  p.b_const.val = 1;
  p.b_const.is_signed = 0;
  p.b_is_const = 1;
  p.rshift_bits = 0;
  p.layer_id = layer_id;
  p.relu_enable = 0;
  ctx.tiu_mul(&p);  // sum_high initialize 0

  p.res_high = nullptr;
  p.res_low = &sum_high;
  p.a = &top;
  p.b_const.val = 0;
  p.b_const.is_signed = 0;
  p.b_is_const = 1;
  p.rshift_bits = 0;
  p.relu_enable = 0;
  ctx.tiu_mul(&p);  // sum_high initialize 0

  for (int step = 1; step <= move_counts && step < input_c; step++) {
    cvk_tdma_l2l_tensor_lrn_shift_param_t lrn_shift_p = {0};
    lrn_shift_p.dst = &shift_sum;
    lrn_shift_p.src = &top;
    lrn_shift_p.right_shift = false;
    lrn_shift_p.lrn_step = step;
    ctx.tdma_l2l_tensor_lrn_shift(&lrn_shift_p);

    cvk_tiu_mac_param_t p3 = {0};
    p3.res_high = &sum_high;
    p3.res_low = &sum;
    p3.res_is_int8 = 0;
    p3.a = &shift_sum;
    p3.b_const.val = 1;
    p3.b_is_const = 1;
    p3.b_const.is_signed = 0;
    p3.lshift_bits = 0;
    p3.rshift_bits = 0;
    p3.layer_id = layer_id;
    p3.relu_enable = 0;
    ctx.tiu_mac(&p3);

    lrn_shift_p.dst = &top_high;
    lrn_shift_p.src = &top;
    lrn_shift_p.right_shift = true;
    lrn_shift_p.lrn_step = step;
    ctx.tdma_l2l_tensor_lrn_shift(&lrn_shift_p);

    p3.res_high = &sum_high;
    p3.res_low = &sum;
    p3.res_is_int8 = 0;
    p3.a = &top_high;
    p3.b_const.val = 1;
    p3.b_is_const = 1;
    p3.b_const.is_signed = 0;
    p3.lshift_bits = 0;
    p3.rshift_bits = 0;
    p3.relu_enable = 0;
    ctx.tiu_mac(&p3);
  }
  // 16bits higher  8bits,
  p.res_high = &top_high;
  p.res_low = &sum_high;
  p.a = &sum_high;
  p.b_const.val = m_i8[0];
  p.b_const.is_signed = 0;
  p.b_is_const = 1;
  p.rshift_bits = 0;
  p.relu_enable = 0;
  ctx.tiu_mul(&p);

  cvk_tiu_mac_param_t p3 = {0};
  p3.res_high = &top_high;
  p3.res_low = &sum_high;
  p3.res_is_int8 = true;
  p3.a = &sum;
  p3.b_const.val = m_i8[0];
  p3.b_is_const = 1;
  p3.b_const.is_signed = 0;
  p3.lshift_bits = 8;
  p3.rshift_bits = sum_rshift_i8;
  p3.layer_id = layer_id;
  p3.relu_enable = 0;
  ctx.tiu_mac(&p3);

  // scale=lut:(k+sum)^(-beta)
  p12.ofmap = &sum_high;
  p12.ifmap = &sum_high;
  p12.table = &pwr_lut;
  p12.layer_id = layer_id;
  ctx.tiu_lookup_table(&p12);

  // Y=x*scale*m_i8[1]>>lrn_rshift_i8
  cvk_tiu_mul_param_t p1 = {0};
  p1.res_high = &top_high;
  p1.res_low = &shift_sum;
  p1.a = &bottom;
  p1.b = &sum_high;
  p1.b_is_const = 0;
  p1.rshift_bits = 0;
  p1.layer_id = layer_id;
  p1.relu_enable = 0;
  ctx.tiu_mul(&p1);

  // 16bits higher  8bits,
  p.res_high = &top_high_high;
  p.res_low = &top_high;
  p.a = &top_high;
  p.b_const.val = m_i8[1];
  p.b_const.is_signed = 0;
  p.b_is_const = 1;
  p.rshift_bits = 0;
  p.relu_enable = 0;
  ctx.tiu_mul(&p);

  p3.res_high = &top_high_high;
  p3.res_low = &top_high;
  p3.res_is_int8 = true;
  p3.a = &shift_sum;
  p3.b_const.val = m_i8[1];
  p3.b_is_const = 1;
  p3.b_const.is_signed = 0;
  p3.lshift_bits = 8;
  p3.rshift_bits = lrn_rshift_i8;
  p3.relu_enable = 0;
  ctx.tiu_mac(&p3);

  cvk_tiu_min_param_t p7 = {0};
  p7.min = &top;
  p7.a = &top_high;
  p7.b_is_const = 1;
  p7.b_const.val = 127;
  p7.b_const.is_signed = 0;
  p7.layer_id = layer_id;
  ctx.tiu_min(&p7);

}


