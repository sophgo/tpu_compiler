/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: lut.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>
#include "backend/backend_tg_api.h"
#include "backend/backend_tl_api.h"


#define DEBUG_TYPE "lut_kernel"


void cvi_backend_tg_fixed_lut_kernel(const CviBackendContext &ctx, uint32_t stream_id,
                                          uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
                                          uint32_t depends_len, gaddr_t bottom_gaddr, gaddr_t top_gaddr,
                                          gaddr_t sg_lut_gaddr, int input_n, int input_c,
                                          int input_h, int input_w, cvk_fmt_t fmt) {

  ctx.set_layer_id(layer_id);

  int table_w, table_h;

  if(fmt == CVK_FMT_I8){
    table_w = 16;
    table_h = 16;
  } else if (fmt == CVK_FMT_BF16) {
    table_w = 8;
    table_h = 32;
  } else {
    assert(0 && "not support type");
  }

  uint8_t eu_align = 1; // hardware constrainst

  // extend to channel
  int require_shape = input_n * input_c * input_h * input_w;
  int coeff_lane_shape = table_h * table_w;
  int blob_num = 1; // 1 means only one blob and it chould overwrite itself
  std::vector<std::pair<cvk_tl_shape_t, gaddr_t>> tiling_info;
  tiling_packing(ctx, require_shape, coeff_lane_shape, blob_num, fmt,
                 &tiling_info);

  cvk_tl_shape_t table_shape = ctx.shape_t4(1, NPU_NUM, table_h, table_w);
  cvk_tl_t *sg_lut_table = ctx.lmem_alloc_tensor(table_shape, fmt, eu_align);

  // load lut table
  ctx.tdma_load_bf16(sg_lut_table, sg_lut_gaddr);
  for (size_t i = 0; i < tiling_info.size(); i++) {
    int n = tiling_info[i].first.n;
    int c = tiling_info[i].first.c;
    int h = tiling_info[i].first.h;
    int w = tiling_info[i].first.w;
    gaddr_t gaddr_offset = tiling_info[i].second;

    // load
    cvk_tl_shape_t input_shape = ctx.shape_t4(n, c, h, w);
    cvk_tl_t *bottom = ctx.lmem_alloc_tensor(input_shape, fmt, eu_align);
    ctx.tdma_load_bf16(bottom, bottom_gaddr + gaddr_offset);

    cvk_tiu_lookup_table_param_t p12 = {0};
    p12.ifmap = bottom;
    p12.ofmap = bottom; // it chould overwrite itself
    p12.table = sg_lut_table;
    p12.layer_id = layer_id;
    ctx.tiu_lookup_table(&p12);

    // move result to global
    ctx.tdma_store_bf16(bottom, top_gaddr + gaddr_offset);

    // free
    ctx.lmem_free_tensor(bottom);
  }
  ctx.lmem_free_tensor(sg_lut_table);
}

static void one_step(const CviBackendContext &ctx, uint32_t layer_id,
                     gaddr_t bottom_gaddr, gaddr_t top_gaddr,
                     cvk_tl_t *tl_table_answer,
                     cvk_tl_t *tl_table_answer_slope,
                     int input_n, int input_c, int input_h, int input_w,
                     gaddr_t gaddr_offset, float scale, float range_min,
                     float range_max, cvk_fmt_t fmt, uint8_t eu_align) {

  cvk_tl_shape_t tl_shape =
      ctx.shape_t4(input_n, input_c, input_h, input_w);

  // tl_ifmap reuse to input / output
  cvk_tl_t *tl_ifmap =
      ctx.lmem_alloc_tensor(tl_shape, fmt, eu_align);
  cvk_tl_t *tl_ofmap_slope =
      ctx.lmem_alloc_tensor(tl_shape, fmt, eu_align);
  cvk_tl_t *tl_ofmap_y0 =
      ctx.lmem_alloc_tensor(tl_shape, fmt, eu_align);

  ctx.tdma_load_bf16(tl_ifmap, bottom_gaddr + gaddr_offset);

#if 0
  // FIXME: enable it after new lut kernel struct merged
  cvk_tiu_bf16_lookup_interp_table_param_t param = {0};
  param.ifmap = tl_ifmap;
  param.buf = tl_ofmap_slope;
  param.tbl_answer = tl_table_answer;
  param.tbl_answer_mantissa = tl_table_answer_slope;
  param.ofmap = tl_ofmap_y0;
  param.is_scientific = 0; // interpolation
  param.min = -8; // align with frontend
  param.max = 8;  // align with frontend
  param.eu_align = eu_align;
  ctx.tiu_bf16_lookup_interp_table(&param);
#else
  cvk_tdma_l2l_tensor_copy_param_t p3 = {0};
  // scale input for remap its idx(-x~x) to (-127~127), dirty tl_ifmap
  cvk_tiu_mul_param_t p4 = {0};
  p4.res_high = NULL;
  p4.res_low = tl_ifmap;
  p4.a = tl_ifmap;
  p4.b_is_const = 1;
  p4.b_const.val = ctx.convert_fp32_to_bf16(scale);
  p4.rshift_bits = 0;
  p4.relu_enable = 0;
  p4.layer_id = layer_id;
  ctx.tiu_mul(&p4);

  // <! get idx from bf16->int8
  memset(&p3, 0x00, sizeof(cvk_tdma_l2l_tensor_copy_param_t));
  cvk_tl_t dst;
  memcpy(&dst, tl_ofmap_y0, sizeof(cvk_tl_t));

//#define L2L_STRIDE
#ifdef L2L_STRIDE
  cvk_tl_shape_t tl_ofmap_x0_int8_shape = {
      1, static_cast<uint32_t>(input_c), static_cast<uint32_t>(input_h * input_w), 1};
  // keep golden
  dst.shape = tl_ofmap_x0_int8_shape;
  dst.fmt = CVK_FMT_I8;
  dst.stride =
      ctx.tl_default_stride(tl_ofmap_x0_int8_shape, CVK_FMT_I8, eu_align);
  dst.stride.h = dst.stride.h * 2;
  dst.int8_rnd_mode = 1;
  p3.dst = &dst;
  p3.src = tl_ifmap;
  ctx.tdma_l2l_bf16_tensor_copy(&p3);
  dst.int8_rnd_mode = 0; // reset

  // <! int8 to bf16 format cus for sub use, sub MUST in the same format
  memset(&p3, 0x00, sizeof(cvk_tdma_l2l_tensor_copy_param_t));
  p3.dst = tl_ofmap_slope; //<! bf16
  p3.src = &dst;
  ctx.tdma_l2l_bf16_tensor_copy(&p3);
#else
  // we keep contiguous layout that we convert
  // it without 'gap' and leverage tiu_mv to
  // contiguous again
  // bf16->int8
  dst.shape = tl_ifmap->shape;
  dst.fmt = CVK_FMT_I8;
  dst.stride = ctx.tl_default_stride(dst.shape, dst.fmt, eu_align);
  dst.int8_rnd_mode = 1;
  p3.src = tl_ifmap;
  p3.dst = &dst;
  ctx.tdma_l2l_bf16_tensor_copy(&p3);
  dst.int8_rnd_mode = 0;

  // int8 to bf16
  p3.src = &dst;
  p3.dst = tl_ofmap_slope; //<! bf16
  ctx.tdma_l2l_bf16_tensor_copy(&p3);
#endif
  // <! sub, diff base , a - b
  // (x - x0)
  cvk_tiu_sub_param_t p5 = {0};
  p5.res_high = 0;
  p5.res_low = tl_ifmap;
  p5.a_high = 0;
  p5.a_low = tl_ifmap;
  p5.b_high = 0;
  p5.b_low = tl_ofmap_slope;
  p5.rshift_bits = 0;
  p5.layer_id = layer_id;
  ctx.tiu_sub(&p5);
#ifdef L2L_STRIDE
#else
  // move index seperate as bf16 size
  // copy to bf16 size
  {
    cvk_tl_t working = *tl_ofmap_slope;
    working.fmt = CVK_FMT_I8;
    cvk_tiu_copy_param_t param = {0};
    param.src = &dst;
    param.dst = &working;
    param.layer_id = layer_id;
    ctx.tiu_copy(&param);
    // back for next index

    dst.fmt = fmt;
    dst.shape = tl_ofmap_slope->shape;
    dst.stride = tl_ofmap_slope->stride;
    param.src = &working;
    param.dst = &dst;
    param.layer_id = layer_id;
    ctx.tiu_copy(&param);
  }
#endif
  // get f(x0) and slope(x)
  // reshape, 16->16
  dst.fmt = fmt;
  dst.shape = tl_ofmap_slope->shape;
  dst.stride = tl_ofmap_slope->stride;

  // <! get slope by index
  cvk_tiu_lookup_table_param_t p6 = {0};
  memset(&p6, 0x0, sizeof(cvk_tiu_lookup_table_param_t));
  p6.ofmap = tl_ofmap_slope;
  p6.ifmap = &dst;
  p6.table = tl_table_answer_slope;
  p6.layer_id = layer_id;
  ctx.tiu_lookup_table(&p6);

  // base f(x0)
  memset(&p6, 0x0, sizeof(cvk_tiu_lookup_table_param_t));
  p6.ofmap = tl_ofmap_y0;
  p6.ifmap = &dst;
  p6.table = tl_table_answer;
  p6.layer_id = layer_id;
  ctx.tiu_lookup_table(&p6);

  // <! mac
  // <! part A + part B, a * b + res = res
  cvk_tiu_mac_param_t p7 = {0};
  p7.res_high = 0;
  p7.res_low = tl_ofmap_y0;
  p7.res_is_int8 = 0;
  p7.a = tl_ifmap;
  p7.b_is_const = 0;
  p7.b = tl_ofmap_slope;
  p7.lshift_bits = 0; // lshift_bits;
  p7.rshift_bits = 0; // rshift_bits;
  p7.relu_enable = 0;
  p7.layer_id = layer_id;
  ctx.tiu_mac(&p7);

#endif

  ctx.tdma_store_bf16(tl_ofmap_y0, top_gaddr + gaddr_offset);
  ctx.lmem_free_tensor(tl_ofmap_y0);
  ctx.lmem_free_tensor(tl_ofmap_slope);
  ctx.lmem_free_tensor(tl_ifmap);
}

void cvi_backend_tg_bf16_lut_interpolation_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id, uint32_t layer_id,
    const uint32_t *depends, uint32_t depends_len, gaddr_t bottom_gaddr,
    gaddr_t top_gaddr, gaddr_t y0_table_gaddr, gaddr_t slope_gaddr, int input_n,
    int input_c, int input_h, int input_w, float range_min, float range_max,
    float scale) {

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "activation_kernel : bottom_gaddr %x top_gaddr %x "
                 "y0_table_gaddr %x slope_gaddr %x "
                 "input_n %d input_c %d input_h %d input_w %d scale %f\n",
                 bottom_gaddr, top_gaddr, y0_table_gaddr, slope_gaddr, input_n,
                 input_c, input_h, input_w, scale));

  // for hw setting
  int const table_n = 1;
  int const table_c = NPU_NUM;
  int const table_h = 32;
  int const table_w = 8;
  uint8_t eu_align = 1; // hardware constrainst

  cvk_tl_shape_t table_shape = {
      static_cast<uint32_t>(table_n), static_cast<uint32_t>(table_c),
      static_cast<uint32_t>(table_h), static_cast<uint32_t>(table_w)};

  cvk_tl_t *tl_table_answer =
      ctx.lmem_alloc_tensor(table_shape, CVK_FMT_BF16, eu_align);
  cvk_tl_t *tl_table_answer_slope =
      ctx.lmem_alloc_tensor(table_shape, CVK_FMT_BF16, eu_align);

  ctx.tdma_load_bf16(tl_table_answer, y0_table_gaddr);
  ctx.tdma_load_bf16(tl_table_answer_slope, slope_gaddr);

  // tiling input
  int blob_num = 3;
  int require_shape = input_n * input_c * input_h * input_w;
  int coeff_lane_shape = 2 * table_h * table_w;
  std::vector<std::pair<cvk_tl_shape_t, gaddr_t>> tiling_info;
  tiling_packing(ctx, require_shape, coeff_lane_shape, blob_num, CVK_FMT_BF16,
                 &tiling_info);

  for (size_t i = 0; i < tiling_info.size(); i++) {
    int n = tiling_info[i].first.n;
    int c = tiling_info[i].first.c;
    int h = tiling_info[i].first.h;
    int w = tiling_info[i].first.w;
    gaddr_t gaddr_offset = tiling_info[i].second;
    one_step(ctx, layer_id, bottom_gaddr, top_gaddr, tl_table_answer,
             tl_table_answer_slope, n, c, h, w, gaddr_offset, scale, range_min,
             range_max, CVK_FMT_BF16, eu_align);
  }

  ctx.lmem_free_tensor(tl_table_answer_slope);
  ctx.lmem_free_tensor(tl_table_answer);
}

void bf16_lut_tl_scientific_forward_kernel(const CviBackendContext &ctx,
    laddr_t la_ifmap,
    laddr_t la_buf,
    laddr_t la_table_answer,
    laddr_t la_table_answer_mantissa,
    laddr_t la_ofmap,
    uint32_t tensor_n, uint32_t tensor_c, uint32_t tensor_h, uint32_t tensor_w,
    uint32_t table_n, uint32_t table_c, uint32_t table_h, uint32_t table_w,
    cvk_fmt_t fmt, uint8_t eu_align) {

  assert(fmt == CVK_FMT_BF16);

  cvk_tl_t tl_ifmap, tl_buf, tl_table_answer, tl_table_answer_mantissa, tl_ofmap;

  // restore tensor from  start_address / shape
  cvi_backend_tl_to_tensor(ctx, &tl_ifmap, la_ifmap,
      tensor_n, tensor_c, tensor_h, tensor_w, fmt, eu_align);
  cvi_backend_tl_to_tensor(ctx, &tl_buf, la_buf,
      tensor_n, tensor_c, tensor_h, tensor_w, fmt, eu_align);
  cvi_backend_tl_to_tensor(ctx, &tl_table_answer, la_table_answer,
      table_n, table_c, table_h, table_w,fmt, eu_align);
  cvi_backend_tl_to_tensor(ctx, &tl_table_answer_mantissa, la_table_answer_mantissa,
      table_n, table_c, table_h, table_w,fmt, eu_align);
  cvi_backend_tl_to_tensor(ctx, &tl_ofmap, la_ofmap,
      tensor_n, tensor_c, tensor_h, tensor_w, fmt, eu_align);

  cvk_tiu_bf16_lookup_interp_table_param_t param = {0};
  param.ifmap = &tl_ifmap;
  param.buf = &tl_buf;
  param.tbl_answer = &tl_table_answer;
  param.tbl_answer_mantissa = &tl_table_answer_mantissa;
  param.ofmap = &tl_ofmap;
#if 1
  // FIXME: enable it after new lut kernel struct merged
  param.is_scientific = 1;
#endif
  ctx.tiu_bf16_lookup_interp_table(&param);
}

void cvi_backend_tg_bf16_lut_scientific_kernel (const CviBackendContext &ctx, uint32_t stream_id,
    uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, gaddr_t bottom_gaddr, gaddr_t top_gaddr,
    gaddr_t exp_lut_table, gaddr_t mantissa_lut_table, int input_n, int input_c,
    int input_h, int input_w, cvk_fmt_t fmt) {

  cvk_tl_shape_t table_shape;
  ctx.bf16_table_shape(&table_shape);

  uint8_t eu_align = 1; // hardware constrainst
  int blob_num = 3; // 3 means we allocate input / output / temp tensor

  // 2 means exponential / mantissa table
  int coeff_lane_shape = 2 * table_shape.h * table_shape.w * sizeof(uint16_t);
  int require_shape = input_n * input_c * input_h * input_w;

  std::vector<std::pair<cvk_tl_shape_t, gaddr_t> > tiling_info;
  tiling_packing(ctx, require_shape, coeff_lane_shape, blob_num, (cvk_fmt_t)fmt, &tiling_info);

  // alloc coeff(table)
  ctx.bf16_table_shape(&table_shape);

  cvk_tl_shape_t cvk_table_shape;
  cvk_table_shape.n = table_shape.n;
  cvk_table_shape.c = table_shape.c;
  cvk_table_shape.h = table_shape.h;
  cvk_table_shape.w = table_shape.w;
  cvk_tl_t *tl_table_answer = ctx.lmem_alloc_tensor(cvk_table_shape, (cvk_fmt_t)fmt, eu_align);
  cvk_tl_t *tl_table_answer_mantissa = ctx.lmem_alloc_tensor(cvk_table_shape, (cvk_fmt_t)fmt, eu_align);

  // load exp / mantissa table
  ctx.tdma_load_bf16(tl_table_answer, exp_lut_table);
  ctx.tdma_load_bf16(tl_table_answer_mantissa, mantissa_lut_table);

  for (size_t i = 0; i < tiling_info.size(); i++) {
    int n = tiling_info[i].first.n;
    int c = tiling_info[i].first.c;
    int h = tiling_info[i].first.h;
    int w = tiling_info[i].first.w;

    gaddr_t gaddr_offset = tiling_info[i].second;

    cvk_tl_shape_t slice_shape = ctx.shape_t4(n, c, h, w);

    // alloc local memory
    cvk_tl_t* tl_ifmap = ctx.lmem_alloc_tensor(slice_shape, (cvk_fmt_t)fmt, eu_align);
    cvk_tl_t* tl_buf = ctx.lmem_alloc_tensor(slice_shape, (cvk_fmt_t)fmt, eu_align);
    cvk_tl_t* tl_ofmap = ctx.lmem_alloc_tensor(slice_shape, (cvk_fmt_t)fmt, eu_align);

    // load input
    ctx.tdma_load_bf16(tl_ifmap, bottom_gaddr + gaddr_offset);
    bf16_lut_tl_scientific_forward_kernel(ctx,
        tl_ifmap->start_address,
        tl_buf->start_address,
        tl_table_answer->start_address,
        tl_table_answer_mantissa->start_address,
        tl_ofmap->start_address,
        n, c, h, w,
        table_shape.n, table_shape.c, table_shape.h, table_shape.w,
        (cvk_fmt_t)fmt, eu_align);

    // TODO checke tfma/tiu pipeline
    // store
    ctx.tdma_store_bf16(tl_ofmap, top_gaddr + gaddr_offset);

    ctx.lmem_free_tensor(tl_ofmap);
    ctx.lmem_free_tensor(tl_buf);
    ctx.lmem_free_tensor(tl_ifmap);
  }

  ctx.lmem_free_tensor(tl_table_answer_mantissa);
  ctx.lmem_free_tensor(tl_table_answer);
}
