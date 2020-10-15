/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: tl_mac_const.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include "backend/backend_tl_api.h"

#define DEBUG_TYPE "tl_mac_const"

void cvi_backend_tl_mac_const(const CviBackendContext &ctx, uint32_t layer_id,
                              gaddr_t input_addr, gaddr_t output_addr,
                              gaddr_t working_addr, int n, int c, int h, int w,
                              int multiplier, int const_val, bool do_relu) {
  // input
  cvk_tl_shape_t tl_shape = ctx.tl_shape_t4(n,c,h,w);
  cvk_tl_t tl_input;
  tl_input.start_address = input_addr;
  tl_input.fmt = CVK_FMT_I8;
  tl_input.shape = tl_shape;
  tl_input.stride = ctx.tl_default_stride(tl_shape, CVK_FMT_I8, 1);

  cvk_tl_t tl_output;
  tl_output.start_address = output_addr;
  tl_output.fmt = CVK_FMT_I8;
  tl_output.shape = tl_shape;
  tl_output.stride = ctx.tl_default_stride(tl_shape, CVK_FMT_I8, 1);

  cvk_tl_t tl_working;
  tl_working.start_address = working_addr;
  tl_working.fmt = CVK_FMT_I8;
  tl_working.shape = tl_shape;
  tl_working.stride = ctx.tl_default_stride(tl_shape, CVK_FMT_I8, 1);

  ctx.parallel_disable();

  cvk_tdma_g2l_tensor_fill_constant_param_t param = {0};
  param.dst = &tl_output;
  param.layer_id = layer_id;
  param.constant = static_cast<int16_t>(const_val);
  ctx.tdma_g2l_tensor_fill_constant(&param);

  cvk_tdma_g2l_tensor_fill_constant_param_t param2 = {0};
  param2.dst = &tl_working;
  param2.layer_id = layer_id;
  param2.constant = static_cast<int16_t>(0);
  ctx.tdma_g2l_tensor_fill_constant(&param2);

  cvk_tiu_mac_param_t p = {0};
  p.res_high = &tl_working;
  p.res_low = &tl_output;
  p.a = &tl_input;
  p.res_is_int8 = 1;
  p.b_const.val = static_cast<int16_t>(multiplier);
  p.b_is_const = 1;
  p.b_const.is_signed = 1;
  p.lshift_bits = 0;
  p.rshift_bits = 0;
  p.layer_id = layer_id;
  p.relu_enable = do_relu ? 1 : 0;
  ctx.tiu_mac(&p);
}

void cvi_backend_bf16_tl_mac_const(const CviBackendContext &ctx,
                                   uint32_t layer_id, laddr_t input_addr,
                                   laddr_t output_addr, int n, int c, int h,
                                   int w, float multiplier, float const_val,
                                   bool do_relu) {
  cvk_tl_shape_t tl_shape = ctx.tl_shape_t4(n,c,h,w);
  cvk_tl_t tl_input;
  tl_input.start_address = input_addr;
  tl_input.fmt = CVK_FMT_BF16;
  tl_input.shape = tl_shape;
  tl_input.stride = ctx.tl_default_stride(tl_shape, CVK_FMT_BF16, 1);

  cvk_tl_t tl_output;
  tl_output.start_address = output_addr;
  tl_output.fmt = CVK_FMT_BF16;
  tl_output.shape = tl_shape;
  tl_output.stride = ctx.tl_default_stride(tl_shape, CVK_FMT_BF16, 1);

  ctx.parallel_disable();

  cvk_tdma_g2l_tensor_fill_constant_param_t param = {0};
  param.dst = &tl_output;
  param.layer_id = layer_id;
  param.constant = ctx.convert_fp32_to_bf16(const_val);
  ctx.tdma_g2l_tensor_fill_constant(&param);

  cvk_tiu_mac_param_t p = {0};
  p.res_high = nullptr;
  p.res_low = &tl_output;
  p.a = &tl_input;
  p.res_is_int8 = 0;
  p.b_const.val = ctx.convert_fp32_to_bf16(multiplier);
  p.b_is_const = 1;
  p.b_const.is_signed = 1;
  p.lshift_bits = 0;
  p.rshift_bits = 0;
  p.layer_id = layer_id;
  p.relu_enable = do_relu ? 1 : 0;
  ctx.tiu_mac(&p);
}
