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
  ctx.tdma_load(*tl_bias, bias_gaddr);
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
