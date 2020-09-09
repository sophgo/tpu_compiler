/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: tensor_common.cpp
 * Description:
 */

#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <cmath>
#include <iostream>
#include "CviBackendContext.h"

#define DEBUG_TYPE "tensor_common"

int is_one_bf16(cvk_fmt_t src, cvk_fmt_t dst) { return is_bf16(src) || is_bf16(dst); }

int is_support_fmt(cvk_fmt_t fmt) {
  assert((fmt == CVK_FMT_I8 || fmt == CVK_FMT_U8 || fmt == CVK_FMT_BF16) && "others not supported");

  return 1;
}

int is_bf16(cvk_fmt_t fmt) { return fmt == CVK_FMT_BF16; }

int _get_csize_local(const CviBackendContext &ctx, int h, int w, cvk_fmt_t fmt) {
  return ___get_lmem_usage(ctx, /*n=*/1, /*c=*/1, h, w, fmt);
}

uint32_t ___get_lmem_usage(const CviBackendContext &ctx, int n, int c, int h, int w, cvk_fmt_t fmt) {
  cvk_tl_shape_t shape = {
      static_cast<uint32_t>(n), static_cast<uint32_t>(c),
      static_cast<uint32_t>(h), static_cast<uint32_t>(w)};
  return ctx.lmem_tensor_to_size(shape, fmt, /*eu_align*/1);
}

int _tensor_size_lmem(const CviBackendContext &ctx, int n, int c, int h, int w, cvk_fmt_t fmt) {
  return n * ALIGN(c, NPU_NUM) * _get_csize_local(ctx, h, w, fmt);
}

int get_csize_local(const CviBackendContext &ctx, int h, int w) {
  return _get_csize_local(ctx, h, w, CVK_FMT_I8);
}

uint32_t __get_lmem_usage(const CviBackendContext &ctx, int n, int c, int h, int w) {
  return ___get_lmem_usage(ctx, n, c, h, w, CVK_FMT_I8);
}

int tensor_size_lmem(const CviBackendContext &ctx, int n, int c, int h, int w) {
  return _tensor_size_lmem(ctx, n, c, h, w, CVK_FMT_I8);
}

void init_tensor_tgmem(const CviBackendContext &ctx, cvk_tg_t *t,
                       uint64_t start_address, cvk_tg_shape_t shape,
                       cvk_tg_stride_t stride, cvk_fmt_t fmt) {
  t->start_address = start_address;
  t->base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(start_address);
  t->fmt = fmt;
  t->shape = shape;
  t->stride = stride;
}

void _tdma_g2g_tensor_copy(const CviBackendContext &ctx, cvk_tg_t *src,
                           cvk_tg_t *dst) {
  cvk_tdma_g2g_tensor_copy_param_t p = {0};
  p.src = src;
  p.dst = dst;

  ctx.tdma_g2g_tensor_copy(&p);
}


void tdma_g2g_tensor_copy(
    // src
    const CviBackendContext &ctx, uint64_t src_start_address,
    cvk_tg_shape_t src_shape, cvk_tg_stride_t src_stride,
    cvk_fmt_t src_fmt,
    // dst
    uint64_t dst_start_address, cvk_tg_shape_t dst_shape,
    cvk_tg_stride_t dst_stride, cvk_fmt_t dst_fmt) {

  cvk_tg_t src;
  src.start_address = src_start_address;
  src.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(src_start_address);
  src.fmt = src_fmt;
  src.shape = src_shape;
  src.stride = src_stride;

  cvk_tg_t dst;
  dst.start_address = dst_start_address;
  dst.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(dst_start_address);
  dst.fmt = dst_fmt;
  dst.shape = dst_shape;
  dst.stride = dst_stride;

  cvk_tdma_g2g_tensor_copy_param_t p = {0};
  p.src = &src;
  p.dst = &dst;

  ctx.tdma_g2g_tensor_copy(&p);
}

/*
int getQuantizeMode(
    const int *i8_multiplier) {
  CviBackendContext::QuantizeMode qmode =
CviBackendContext::QuantizeMode::INT8_NOTSUPPORT; if (i8_multiplier) { qmode =
CviBackendContext::QuantizeMode::INT8_PER_LAYER;
  }
  else {
    qmode = CviBackendContext::QuantizeMode::INT8_32_MULTIPLER;
  }

  assert(qmode != CviBackendContext::QuantizeMode::INT8_NOTSUPPORT);
  return qmode;
}
*/

/**
 * \brief load bias(16bytes) or 32 bit multiplier, cus multiplier store int 'bias'
 * \tl_bias set nullptr once INT8_PER_LAYER && !do_bias
 */
void load_bias_multiplier(const CviBackendContext &ctx,
                          int oc_step,  // output channel
                          bool do_bias, gaddr_t bias_gaddr, int qmode,
                          cvk_tl_t **tl_bias) {

  if (qmode == CviBackendContext::QuantizeMode::INT8_32_MULTIPLER) {
    load_32byte_multiplier(ctx, oc_step, do_bias, bias_gaddr, tl_bias);
  } else if (qmode == CviBackendContext::QuantizeMode::INT8_PER_LAYER) {
    if (do_bias) {
      load_16bytes_bias(ctx, oc_step, tl_bias, bias_gaddr);
    } else {
      *tl_bias = nullptr;
    }
  }
}

void load_32byte_multiplier(const CviBackendContext &ctx, int oc_step, bool do_bias,
                            gaddr_t bias_gaddr,
                            cvk_tl_t **tl_chl_quan_param  // 32 byte multiplier
  ) {


  if (do_bias) {
    // call ctx.tl_default_stride
    cvk_tl_shape_t coeff_shape_9byte = ctx.shape_t4(1, oc_step, 1, 9);
    *tl_chl_quan_param = ctx.lmem_alloc_tensor(coeff_shape_9byte, CVK_FMT_U8, /*eu_align=*/0);
  } else {
    cvk_tl_shape_t coeff_shape_5byte = ctx.shape_t4(1, oc_step, 1, 5);
    *tl_chl_quan_param = ctx.lmem_alloc_tensor(coeff_shape_5byte, CVK_FMT_U8, /*eu_align=*/0);
  }

  ctx.tdma_load(*tl_chl_quan_param, bias_gaddr);
}

void load_16bytes_bias(const CviBackendContext &ctx, int oc, cvk_tl_t **tl_bias,
                       gaddr_t bias_gaddr) {

  cvk_tl_shape_t tl_bias_shape;
  tl_bias_shape.n = 2;
  tl_bias_shape.c = oc;
  tl_bias_shape.h = 1;
  tl_bias_shape.w = 1;
  *tl_bias = ctx.lmem_alloc_tensor(tl_bias_shape, CVK_FMT_I8, 0);

  cvk_tg_t ts_bias;
  ts_bias.start_address = bias_gaddr;
  ts_bias.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(bias_gaddr);
  ts_bias.fmt = CVK_FMT_I8;
  ts_bias.shape.n = 2;
  ts_bias.shape.c = oc;
  ts_bias.shape.h = 1;
  ts_bias.shape.w = 1;
  ts_bias.stride = ctx.tg_default_stride(ts_bias.shape,ts_bias.fmt);

  cvk_tdma_g2l_tensor_copy_param_t p = {0};
  p.src = &ts_bias;
  p.dst = *tl_bias;
  ctx.tdma_g2l_tensor_copy(&p);
}

// copy same shape from system to local
void tdma_g2l_tensor_copy(const CviBackendContext &ctx, cvk_tl_t **tl_bslice,
                          int input_n, int input_c, int input_h, int input_w, gaddr_t input_gaddr,
                          cvk_fmt_t fmt, int eu_align) {

  cvk_tl_shape_t tl_bslice_shape;
  tl_bslice_shape.n = input_n;
  tl_bslice_shape.c = input_c;
  tl_bslice_shape.h = input_h;
  tl_bslice_shape.w = input_w;

  *tl_bslice = ctx.lmem_alloc_tensor(tl_bslice_shape, fmt, eu_align);

  cvk_tg_t ts_bslice;
  ts_bslice.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(input_gaddr);
  ts_bslice.fmt = fmt;
  ts_bslice.start_address = input_gaddr;
  ts_bslice.shape.n = tl_bslice_shape.n;
  ts_bslice.shape.c = tl_bslice_shape.c;
  ts_bslice.shape.h = tl_bslice_shape.h;
  ts_bslice.shape.w = tl_bslice_shape.w;
  ts_bslice.stride = ctx.tg_default_stride(ts_bslice.shape, ts_bslice.fmt);

  cvk_tdma_g2l_tensor_copy_param_t p = {0};
  p.src = &ts_bslice;
  p.dst = *tl_bslice;
  ctx.tdma_g2l_tensor_copy(&p);
}

// apply quantize int 8 mode
void apply_qi8(const CviBackendContext &ctx, cvk_tl_t *ifmap, uint32_t layer_id, int do_relu,
               int right_shift_width, int threshold_x_quantized) {

  cvk_tiu_mul_param_t p = {0};
  p.res_high = nullptr;
  p.res_low = ifmap;
  p.a = ifmap;
  p.b_const.val = threshold_x_quantized;
  p.b_const.is_signed = false;
  p.b_is_const = 1;
  p.rshift_bits = right_shift_width;
  p.layer_id = layer_id;
  p.relu_enable = do_relu;
  ctx.tiu_mul(&p);
}

void *cvi_backend_get_cvk_ctx(const CviBackendContext &ctx) {
  return ctx.get_cvk_ctx();
}

cvk_fmt_t cvi_to_cvk_fmt(cvi_backend_fmt_t cvi_backend_fmt) {
  switch (cvi_backend_fmt) {
    case CVI_FMT_F32:   return CVK_FMT_F32;
    case CVI_FMT_F16:   return CVK_FMT_F16;
    case CVI_FMT_I32:   return CVK_FMT_I32;
    case CVI_FMT_I16:   return CVK_FMT_I16;
    case CVI_FMT_I8:    return CVK_FMT_I8;
    case CVI_FMT_I4:    return CVK_FMT_I4;
    case CVI_FMT_I2:    return CVK_FMT_I2;
    case CVI_FMT_I1:    return CVK_FMT_I1;
    case CVI_FMT_U32:   return CVK_FMT_U32;
    case CVI_FMT_U16:   return CVK_FMT_U16;
    case CVI_FMT_U8:    return CVK_FMT_U8;
    case CVI_FMT_BF16:  return CVK_FMT_BF16;
    default: assert(0 && "not support type in cvi_backend_fmt_t");
  }
  return CVK_FMT_INVALID;
}
