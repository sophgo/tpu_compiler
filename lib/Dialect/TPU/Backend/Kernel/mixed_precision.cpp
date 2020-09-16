/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: mixed_precision.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>

#define DEBUG_TYPE "mixed_precision"

/**
 * \brief bf16 quant to uint8_t/i8 depends \fmt_d, quant means narrow down from wise precision
 * shrink to narrow one
 * ss means system(ddr) to system, however, we implement by load->mul->store it
 * \TODO merge to eltwise prod
 * \fmt_s source format
 */
static void _mixed_precision_ss_bf16_quant(const CviBackendContext &ctx, uint32_t stream_id,
    uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, gaddr_t bottom_gaddr, gaddr_t top_gaddr,
    int input_n, int input_c, int input_h, int input_w , float const_scale,
    cvk_fmt_t fmt_s, cvk_fmt_t fmt_d
    ) {

  assert(fmt_s == CVK_FMT_BF16 || fmt_s == CVK_FMT_F32);
  assert(fmt_d == CVK_FMT_I8 || fmt_d == CVK_FMT_U8);

  // in fp32->int8 case, we need to allocate size of unit bf16
  // and convert(bf16->int8) itself
  cvk_fmt_t tiling_unit = CVK_FMT_BF16;
  // 2 means fp32 takes twice size than bf16
  int unit_size = fmt_s == CVK_FMT_BF16 ? 1 : 2;
  int blob_num = 1;

  int coeff_lane_shape = 0;
  if (fmt_s == CVK_FMT_F32) {
    // * 2 means fp32 takes twice size of bf16
    blob_num = 2;
    coeff_lane_shape = 2; // 2 means reserve 2 byte fp32->bf16
  }

  int require_shape = input_n * input_c * input_h * input_w;

  std::vector<std::pair<cvk_tl_shape_t, gaddr_t> > tiling_info;
  tiling_packing(ctx, require_shape, coeff_lane_shape, blob_num, tiling_unit, &tiling_info);

  for (size_t i = 0; i < tiling_info.size(); i++) {
    int n = tiling_info[i].first.n;
    int c = tiling_info[i].first.c;
    int h = tiling_info[i].first.h;
    int w = tiling_info[i].first.w;

    gaddr_t gaddr_offset = tiling_info[i].second;

    cvk_tl_shape_t slice_shape = ctx.shape_t4(n, c, h, w);

    if (fmt_s == CVK_FMT_F32) {
      // contiguous load, directly load fp32
      slice_shape = ctx.shape_t4(n, c, h, w * 2);
    }

    // alloc local memory
    cvk_tl_t* tl_ifmap = ctx.lmem_alloc_tensor(slice_shape, tiling_unit, /*eu_align=*/0);

    // load input
    // FIXME: use \cvi_backend_tl_load_tensor
    //cvi_backend_tl_load_tensor(ctx, layer_id, tl_ifmap, bottom_gaddr + gaddr_offset, /*eu_align=*/0);
    gaddr_t bottom_offset = bottom_gaddr + gaddr_offset * unit_size;
    ctx.tdma_load_bf16(tl_ifmap, bottom_offset);

    // quant it
    cvk_tiu_mul_param_t p = {0};
    p.res_high = NULL;
    p.res_low = tl_ifmap;
    p.a = tl_ifmap;
    p.b_is_const = 1;
    p.b_const.val = ctx.convert_fp32_to_bf16(const_scale);
    p.relu_enable = 0;
    p.layer_id = layer_id;
    ctx.tiu_mul(&p);


    laddr_t tl_ifmap_laddr = tl_ifmap->start_address;
    if (fmt_s == CVK_FMT_F32) {
      cvk_tl_t tl_ifmap_fp32 = *tl_ifmap;
      lmem_shrink_fp32_bf16(ctx, tl_ifmap, &tl_ifmap_fp32, n, c, h, w, layer_id);
    }

    // FIXME: leverage \tdma_store_stride_bf16 with pass \fmt_d
    cvk_tg_t ts_data;
    ts_data.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(top_gaddr);
    ts_data.fmt = fmt_d;
    // 2 means bf16 size is 2 times than int8
    ts_data.start_address = top_gaddr + (gaddr_offset / 2);
    ts_data.shape.n = tl_ifmap->shape.n;
    ts_data.shape.c = tl_ifmap->shape.c;
    ts_data.shape.h = tl_ifmap->shape.h;
    ts_data.shape.w = tl_ifmap->shape.w;
    ts_data.stride = ctx.tg_default_stride(ts_data.shape, fmt_d);

    // emit, transform from bf16 to int8
    cvk_tdma_l2g_tensor_copy_param_t p1 = {0};
    p1.src = tl_ifmap;
    p1.dst = &ts_data;
    ctx.tdma_l2g_bf16_tensor_copy(&p1);

    tl_ifmap->start_address = tl_ifmap_laddr;
    ctx.lmem_free_tensor(tl_ifmap);
  }
}

/*
 * \brief dequant means from narrow one extend to large one
 */
static void _mixed_precision_ss_8bit_dequant(const CviBackendContext &ctx, uint32_t stream_id,
    uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, gaddr_t bottom_gaddr, gaddr_t top_gaddr,
    int input_n, int input_c, int input_h, int input_w , float const_scale,
    cvk_fmt_t fmt_s, cvk_fmt_t fmt_d
    ) {

  assert(fmt_s == CVK_FMT_I8 || fmt_s == CVK_FMT_U8);
  assert(fmt_d == CVK_FMT_BF16 || fmt_d == CVK_FMT_F32);

  int blob_num = 1; // bf16 only
  int coeff_lane_shape = 0;
  int unit_size = 1;
  if (fmt_d == CVK_FMT_F32) {
    blob_num = 3; // 2 means fp32 take twice than bf16
    // +2 means we prevent wrap to top, reserver it
    coeff_lane_shape = 2;
    // output size if fp32 that takes twice size than bf16
    unit_size = 2;
  }

  cvk_fmt_t tiling_unit = CVK_FMT_BF16;
  cvk_fmt_t ofmap_fmt = fmt_d;

  if (fmt_d == CVK_FMT_F32) {
    // just use bf16 size and store back
    ofmap_fmt = CVK_FMT_BF16;
    // clean output lmem to 0
    fill_fp32_lmem_0(ctx, layer_id, input_n, input_c, input_h, input_w);
  }

  int require_shape = input_n * input_c * input_h * input_w;

  std::vector<std::pair<cvk_tl_shape_t, gaddr_t> > tiling_info;
  tiling_packing(ctx, require_shape, coeff_lane_shape, blob_num, tiling_unit, &tiling_info);

  for (size_t i = 0; i < tiling_info.size(); i++) {
    uint32_t n = tiling_info[i].first.n;
    uint32_t c = tiling_info[i].first.c;
    uint32_t h = tiling_info[i].first.h;
    uint32_t w = tiling_info[i].first.w;

    gaddr_t gaddr_offset = tiling_info[i].second;

    cvk_tl_shape_t slice_shape = ctx.shape_t4(n, c, h, w);

    // output fp32 used, it takes twice size than bf16
    cvk_tl_shape_t output_shape = ctx.shape_t4(n, c, h, w * 2);

    // alloc local memory
    cvk_tl_t* tl_ofmap_fp32 = NULL;
    cvk_tl_t* tl_gap = NULL;
    laddr_t tl_ofmap_laddr = LA_INVALID;

    if (fmt_d == CVK_FMT_F32) {
      tl_ofmap_fp32 = ctx.lmem_alloc_tensor(output_shape, ofmap_fmt, /*eu_align=*/0);
      tl_ofmap_laddr = tl_ofmap_fp32->start_address;

      cvk_tl_shape_t gap_shape = ctx.shape_t4(1, c, 1, coeff_lane_shape);
      tl_gap = ctx.lmem_alloc_tensor(gap_shape, ofmap_fmt, /*eu_align=*/0);

      tl_ofmap_fp32->start_address = tl_ofmap_laddr + 2;// +2 means put start point at higher 16bit
      tl_ofmap_fp32->shape = slice_shape; // fake
      tl_ofmap_fp32->stride = tl_fp32_stride(ctx, tl_ofmap_fp32);
    }

    cvk_tl_t* tl_ofmap = ctx.lmem_alloc_tensor(slice_shape, ofmap_fmt, /*eu_align=*/0);
    cvk_tl_t* output = tl_ofmap;

    // load input
    // i8->bf16
    cvk_tdma_g2l_tensor_copy_param_t p1 = {0};
    cvk_tg_t src;
    // half for src type is i8, we calculate shape with bf16
    src.start_address = bottom_gaddr + gaddr_offset / 2;
    src.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(bottom_gaddr + gaddr_offset);
    src.int8_rnd_mode = 0;
    src.fmt = fmt_s;
    src.shape = {n, c, h, w};
    src.stride = ctx.tg_default_stride(src.shape, src.fmt);

    p1.src = &src;
    p1.dst = tl_ofmap;
    ctx.tdma_g2l_bf16_tensor_copy(&p1);

    // mul scale(dequant)
    cvk_tiu_mul_param_t p = {0};
    p.res_high = NULL;
    p.res_low = (fmt_d == CVK_FMT_F32) ? tl_ofmap_fp32 : tl_ofmap;
    p.a = tl_ofmap;
    p.b_is_const = 1;
    p.b_const.val = ctx.convert_fp32_to_bf16(const_scale);
    p.relu_enable = 0;
    p.layer_id = layer_id;

    ctx.tiu_mul(&p);

    // emit
    if (fmt_d == CVK_FMT_F32) {
      tl_ofmap_fp32->start_address = tl_ofmap_laddr;
      tl_ofmap_fp32->shape = output_shape;
      tl_ofmap_fp32->stride = ctx.tl_default_stride(tl_ofmap_fp32->shape, tl_ofmap_fp32->fmt, /*eu_align=*/0);
      output = tl_ofmap_fp32;
    }

    ctx.tdma_store_bf16(output, top_gaddr + gaddr_offset * unit_size);

    // release
    ctx.lmem_free_tensor(tl_ofmap);
    if (tl_gap) {
      ctx.lmem_free_tensor(tl_gap);
    }
    if (tl_ofmap_fp32) {
      ctx.lmem_free_tensor(tl_ofmap_fp32);
    }
  }
}
/**
 * \brief load from tg(tensor global, sys) to tl(tensor local, local)
 * with bf16 format and store back as uint8_t to tg
 */
void mixed_precision_tg_bf16_u8(const CviBackendContext &ctx, uint32_t stream_id,
    uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, gaddr_t bottom_gaddr, gaddr_t top_gaddr,
    int input_n, int input_c, int input_h, int input_w, float const_scale
    ) {

  _mixed_precision_ss_bf16_quant(ctx, stream_id,
      inst_id, layer_id, depends,
      depends_len, bottom_gaddr, top_gaddr,
      input_n, input_c, input_h, input_w, const_scale,
      CVK_FMT_BF16, CVK_FMT_U8);
}


/**
 * \brief load from tg(tensor global, sys) to tl(tensor local, local)
 * with bf16 format and store back as s8 to tg
 */
void mixed_precision_tg_bf16_s8(const CviBackendContext &ctx, uint32_t stream_id,
    uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, gaddr_t bottom_gaddr, gaddr_t top_gaddr,
    int input_n, int input_c, int input_h, int input_w, float const_scale
    ) {

  _mixed_precision_ss_bf16_quant(ctx, stream_id,
      inst_id, layer_id, depends,
      depends_len, bottom_gaddr, top_gaddr,
      input_n, input_c, input_h, input_w, const_scale,
      CVK_FMT_BF16, CVK_FMT_I8);
}

void mixed_precision_quant(const CviBackendContext &ctx,
    uint32_t layer_id,
    cvk_fmt_t from, cvk_fmt_t to,
    gaddr_t bottom_gaddr, gaddr_t top_gaddr,
    int input_n, int input_c, int input_h, int input_w, float const_scale
    ) {

  _mixed_precision_ss_bf16_quant(ctx, 0,
      0, layer_id, NULL,
      0, bottom_gaddr, top_gaddr,
      input_n, input_c, input_h, input_w, const_scale,
      from, to);
}

/**
 * \brief load from tg(tensor global, sys) to tl(tensor local, local)
 * with uint8_t format and store back as bf16 to tg
 */
void mixed_precision_tg_u8_bf16(const CviBackendContext &ctx, uint32_t stream_id,
    uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, gaddr_t bottom_gaddr, gaddr_t top_gaddr,
    int input_n, int input_c, int input_h, int input_w, float const_scale
    ) {

  _mixed_precision_ss_8bit_dequant(ctx, stream_id,
      inst_id, layer_id, depends,
      depends_len, bottom_gaddr, top_gaddr,
      input_n, input_c, input_h, input_w, const_scale,
      CVK_FMT_U8, CVK_FMT_BF16);
}

/**
 * \brief load from tg(tensor global, sys) to tl(tensor local, local)
 * with s8 format and store back as bf16 to tg
 */
void mixed_precision_tg_s8_bf16(const CviBackendContext &ctx, uint32_t stream_id,
    uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
    uint32_t depends_len, gaddr_t bottom_gaddr, gaddr_t top_gaddr,
    int input_n, int input_c, int input_h, int input_w, float const_scale
    ) {

  _mixed_precision_ss_8bit_dequant(ctx, stream_id,
      inst_id, layer_id, depends,
      depends_len, bottom_gaddr, top_gaddr,
      input_n, input_c, input_h, input_w, const_scale,
      CVK_FMT_I8, CVK_FMT_BF16);
}

void mixed_precision_dequant(const CviBackendContext &ctx,
    uint32_t layer_id,
    cvk_fmt_t from, cvk_fmt_t to,
    gaddr_t bottom_gaddr, gaddr_t top_gaddr,
    int input_n, int input_c, int input_h, int input_w, float const_scale
    ) {

  _mixed_precision_ss_8bit_dequant(ctx, 0,
      0, layer_id, NULL,
      0, bottom_gaddr, top_gaddr,
      input_n, input_c, input_h, input_w, const_scale,
      from, to);
}
