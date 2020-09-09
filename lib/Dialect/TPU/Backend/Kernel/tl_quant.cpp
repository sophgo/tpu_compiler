/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: tl_quant.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include "backend/backend_tl_api.h"

#define DEBUG_TYPE "tl_quant"

// for layer group

void cvi_backend_tl_quant(
    const CviBackendContext &ctx, uint32_t layer_id,
    laddr_t la_input, laddr_t la_output,
    cvk_fmt_t from, cvk_fmt_t to,
    float const_scale,
    int n, int c, int h, int w) {

  ctx.set_layer_id(layer_id);
  LLVM_DEBUG(
    llvm::errs() << "    Quant  : nchw = ("
                 << n << "," << c << "," << h << "," << w << ")"
                 << "\n                     "
                 << "la_i = " << la_input
                 << ", la_o = " << la_output
                 << "\n";
  );
  assert((from == CVK_FMT_I8 || from == CVK_FMT_U8 || from == CVK_FMT_BF16) &&
         "`from` only support int8/bf16");
  assert((to == CVK_FMT_I8 || to == CVK_FMT_BF16) && "`to` only support int8/bf16");
  assert((from != to) && "`from` and `to` not equal");

  int is_dequant =
      ((from == CVK_FMT_I8 || from == CVK_FMT_U8) && to == CVK_FMT_BF16);

  cvk_tl_t *tl_input = new cvk_tl_t;
  cvk_tl_t *tl_output = new cvk_tl_t;
  cvk_fmt_t _from, _to;

  _from = cvi_to_cvk_fmt(from);
  _to = cvi_to_cvk_fmt(to);

  tl_input->start_address = la_input;
  tl_input->fmt = _from;
  tl_input->shape = ctx.shape_t4(n, c, h, w);
  tl_input->stride = ctx.tl_default_stride(tl_input->shape, tl_input->fmt, /*eu_align=*/1);

  tl_output->start_address = la_output;
  tl_output->fmt = _to;
  tl_output->shape = ctx.shape_t4(n, c, h, w);
  tl_output->stride = ctx.tl_default_stride(tl_output->shape, tl_output->fmt, /*eu_align=*/1);

  //compute
  if (is_dequant) {
    cvk_tdma_l2l_tensor_copy_param_t p1 = {0};
    p1.src = tl_input;
    p1.dst = tl_output;
    ctx.tdma_l2l_bf16_tensor_copy(&p1);

    // NOTICE: make sure tdma order before than mul
    ctx.parallel_enable();
    ctx.parallel_disable();

    // move to high accurcy to calculate quant/dequant
    cvk_tiu_mul_param_t p = {0};
    p.res_high = NULL;
    p.res_low = tl_output;
    p.a = tl_output;
    p.b_is_const = 1;
    p.b_const.val = ctx.convert_fp32_to_bf16(const_scale);
    p.relu_enable = 0;
    p.layer_id = layer_id;

    ctx.tiu_mul(&p);
  }
  else {
    // quant, bf16->int8
    // move to high accurcy to calculate quant/dequant
    cvk_tiu_mul_param_t p = {0};
    p.res_high = NULL;
    p.res_low = tl_input;
    p.a = tl_input;
    p.b_is_const = 1;
    p.b_const.val = ctx.convert_fp32_to_bf16(const_scale);
    p.relu_enable = 0;
    p.layer_id = layer_id;

    ctx.tiu_mul(&p);

    // NOTICE: make sure tdma order before than mul
    ctx.parallel_enable();
    ctx.parallel_disable();

    // leverage l2l to implement bf16->int8
    cvk_tdma_l2l_tensor_copy_param_t p1 = {0};
    p1.src = tl_input;
    p1.dst = tl_output;
    ctx.tdma_l2l_bf16_tensor_copy(&p1);
  }

  delete tl_output;
  delete tl_input;

}
