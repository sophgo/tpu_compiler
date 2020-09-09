/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: tl_crop.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include "backend/backend_tl_api.h"
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#define DEBUG_TYPE "tl_crop"

void cvi_backend_tl_crop(const CviBackendContext &ctx, uint32_t layer_id,
                         int64_t *input_dim, int64_t *output_dim, laddr_t la_input,
                         laddr_t la_output, int *offsets) {

  LLVM_DEBUG(llvm::errs() << llvm::format("cvi_backend_tl_crop:\n"
                                          "  layer_id %d\n",
                                          layer_id));

  LLVM_DEBUG(llvm::errs() << llvm::format("la_input:%d\n", la_input));

  LLVM_DEBUG(llvm::errs() << llvm::format("la_output:%d\n", la_output));

  ctx.parallel_disable();

  uint32_t in = input_dim[0];
  uint32_t ic = input_dim[1];
  uint32_t ih = input_dim[2];
  uint32_t iw = input_dim[3];

  uint32_t on = output_dim[0];
  uint32_t oc = output_dim[1];
  uint32_t oh = output_dim[2];
  uint32_t ow = output_dim[3];

  uint32_t offset_n = offsets[0];
  uint32_t offset_c = offsets[1];
  uint32_t offset_h = offsets[2];
  uint32_t offset_w = offsets[3];

  cvk_tl_shape_t input_shape = {in, ic, ih, iw};

  cvk_tl_shape_t output_shape = {on, oc, oh, ow};
  cvk_tl_t tl_input;
  tl_input.start_address = la_input;
  tl_input.fmt = CVK_FMT_I8;
  tl_input.shape = input_shape;
  tl_input.stride = ctx.tl_default_stride(input_shape, CVK_FMT_I8, 1);

  cvk_tl_t tl_output;
  tl_output.start_address = la_output;
  tl_output.fmt = CVK_FMT_I8;
  tl_output.shape = output_shape;
  tl_output.stride = ctx.tl_default_stride(output_shape, CVK_FMT_I8, 1);

  auto input_offset = offset_n * tl_input.stride.n +
                      ceiling_func(offset_c, NPU_NUM) * tl_input.stride.c +
                      offset_h * tl_input.stride.h + offset_w * tl_input.stride.w;

  uint32_t input_addr = la_input + input_offset;
  tl_input.start_address = input_addr;
  tl_input.shape = output_shape;
  cvk_tdma_l2l_tensor_copy_param_t p1 = {0};
  p1.dst = &tl_output;
  p1.src = &tl_input;
  ctx.tdma_l2l_tensor_copy(&p1);
}

void cvi_backend_tl_bf16_crop(const CviBackendContext &ctx, uint32_t layer_id,
                              int64_t *input_dim, int64_t *output_dim, laddr_t la_input,
                              laddr_t la_output, int *offsets) {

  LLVM_DEBUG(llvm::errs() << llvm::format("cvi_backend_tl_bf16_crop:\n"
                                          "  layer_id %d\n",
                                          layer_id));

  LLVM_DEBUG(llvm::errs() << llvm::format("la_input:%d\n", la_input));

  LLVM_DEBUG(llvm::errs() << llvm::format("la_output:%d\n", la_output));

  ctx.parallel_disable();

  uint32_t in = input_dim[0];
  uint32_t ic = input_dim[1];
  uint32_t ih = input_dim[2];
  uint32_t iw = input_dim[3];

  uint32_t on = output_dim[0];
  uint32_t oc = output_dim[1];
  uint32_t oh = output_dim[2];
  uint32_t ow = output_dim[3];

  uint32_t offset_n = offsets[0];
  uint32_t offset_c = offsets[1];
  uint32_t offset_h = offsets[2];
  uint32_t offset_w = offsets[3];

  cvk_tl_shape_t input_shape = {in, ic, ih, iw};
  cvk_tl_shape_t output_shape = {on, oc, oh, ow};
  cvk_tl_t tl_input;
  tl_input.start_address = la_input;
  tl_input.fmt = CVK_FMT_BF16;
  tl_input.shape = input_shape;
  tl_input.stride = ctx.tl_default_stride(input_shape, CVK_FMT_BF16, 1);

  cvk_tl_t tl_output;
  tl_output.start_address = la_output;
  tl_output.fmt = CVK_FMT_BF16;
  tl_output.shape = output_shape;
  tl_output.stride = ctx.tl_default_stride(output_shape, CVK_FMT_BF16, 1);

  auto input_offset = offset_n * tl_input.stride.n +
                      ceiling_func(offset_c, NPU_NUM) * tl_input.stride.c +
                      offset_h * tl_input.stride.h + offset_w * tl_input.stride.w;

  uint32_t input_addr = la_input + input_offset;
  tl_input.start_address = input_addr;
  tl_input.shape = output_shape;
  cvk_tdma_l2l_tensor_copy_param_t p1 = {0};
  p1.dst = &tl_output;
  p1.src = &tl_input;
  ctx.tdma_l2l_bf16_tensor_copy(&p1);
}