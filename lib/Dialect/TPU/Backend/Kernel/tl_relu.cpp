/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: tl_pad.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include "backend/backend_tl_api.h"
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#define DEBUG_TYPE "tl_relu"

void cvi_backend_tl_relu(const CviBackendContext &ctx, uint32_t layer_id,
                         int n, int c, int h, int w,
                         laddr_t la_input, laddr_t la_output) {

  LLVM_DEBUG(
        llvm::errs() << llvm::format("cvi_backend_tl_relu:\n"
                                  "  layer_id %d\n", layer_id));
  LLVM_DEBUG(
        llvm::errs() << llvm::format("la_input:%d\n", la_input));

  LLVM_DEBUG(
        llvm::errs() << llvm::format("la_output:%d\n", la_output));

  cvk_tl_shape_t shape = {static_cast<uint32_t>(n),
                          static_cast<uint32_t>(c),
                          static_cast<uint32_t>(h),
                          static_cast<uint32_t>(w)};
  cvk_tl_t tl_input;
  tl_input.start_address = la_input;
  tl_input.fmt = CVK_FMT_I8;
  tl_input.shape = shape;
  tl_input.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);

  cvk_tl_t tl_output;
  tl_output.start_address = la_output;
  tl_output.fmt = CVK_FMT_I8;
  tl_output.shape = shape;
  tl_output.stride = ctx.tl_default_stride(shape, CVK_FMT_I8, 1);

  cvk_tiu_max_param_t p1 = {0};
  p1.max = &tl_output;
  p1.a = &tl_input;
  p1.b_is_const = 1;
  p1.b_const.is_signed = 1;
  p1.b_const.val = 0;
  p1.layer_id = layer_id;
  ctx.tiu_max(&p1);
 }


void cvi_backend_tl_bf16_relu(const CviBackendContext &ctx,
                              uint32_t layer_id,
                              int n, int c, int h, int w,
                              laddr_t la_input, laddr_t la_output) {

  LLVM_DEBUG(
        llvm::errs() << llvm::format("cvi_backend_tl_bf16_relu:\n"
                                  "  layer_id %d\n", layer_id));
  LLVM_DEBUG(
        llvm::errs() << llvm::format("la_input:%d\n", la_input));

  LLVM_DEBUG(
        llvm::errs() << llvm::format("la_output:%d\n", la_output));

  cvk_tl_shape_t shape = {static_cast<uint32_t>(n),
                          static_cast<uint32_t>(c),
                          static_cast<uint32_t>(h),
                          static_cast<uint32_t>(w)};
  cvk_tl_t tl_input;
  tl_input.start_address = la_input;
  tl_input.fmt = CVK_FMT_BF16;
  tl_input.shape = shape;
  tl_input.stride = ctx.tl_default_stride(shape, CVK_FMT_BF16, 1);

  cvk_tl_t tl_output;
  tl_output.start_address = la_output;
  tl_output.fmt = CVK_FMT_BF16;
  tl_output.shape = shape;
  tl_output.stride = ctx.tl_default_stride(shape, CVK_FMT_BF16, 1);

  cvk_tiu_max_param_t p1 = {0};
  p1.max = &tl_output;
  p1.a = &tl_input;
  p1.b_is_const = 1;
  p1.b_const.is_signed = 1;
  p1.b_const.val = 0;
  p1.layer_id = layer_id;
  ctx.tiu_max(&p1);
 }