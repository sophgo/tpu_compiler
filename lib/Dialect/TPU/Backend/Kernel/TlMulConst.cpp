/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: tl_mul_const.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include "backend/backend_tl_api.h"
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#define DEBUG_TYPE "tl_mul_const"

void cvi_backend_tl_mul_const(const CviBackendContext &ctx, uint32_t layer_id,
                              laddr_t la_input, laddr_t la_output,
                              float const_val, bool do_relu, int n, int c,
                              int h, int w, cvk_fmt_t fmt) {

  cvk_tl_shape_t shape = ctx.tl_shape_t4(n, c, h, w);
  cvk_tl_t bottom = {0};
  cvk_tl_t top = {0};
  ctx.lmem_init_tensor(&bottom, shape, fmt, 1);
  ctx.lmem_init_tensor(&top, shape, fmt, 1);
  bottom.start_address = la_input;
  top.start_address = la_output;
  uint16_t val;
  if (fmt == CVK_FMT_BF16) {
    val = ctx.convert_fp32_to_bf16(const_val);
  } else {
    val = static_cast<int8_t>(const_val);
  }
  if (const_val == 1.0f && do_relu == false) {
    cvk_tiu_copy_param_t p = {0};
    p.src = &bottom;
    p.dst = &top;
    p.layer_id = layer_id;
    ctx.tiu_copy(&p);
  } else {
    cvk_tiu_mul_param_t p1 = {0};
    p1.res_high = nullptr;
    p1.res_low = &top;
    p1.a = &bottom;
    p1.b_const.val = val;
    p1.b_const.is_signed = true;
    p1.b_is_const = 1;
    p1.rshift_bits = 0;
    p1.layer_id = layer_id;
    p1.relu_enable = do_relu;
    ctx.tiu_mul(&p1);
  }
}
