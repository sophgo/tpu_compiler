/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: tl_concat.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include "backend/backend_tl_api.h"
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#define DEBUG_TYPE "tl_concat"

// axis = 1
void cvi_backend_tl_concat(const CviBackendContext &ctx, uint32_t layer_id,
                           int *input_dim_c, int input_size, int *output_dim,
                           laddr_t *la_input, laddr_t la_output, laddr_t la_working,
                           bool do_relu, int8_t rshift, int8_t * m_i8) {

LLVM_DEBUG(
      llvm::errs() << llvm::format("cvi_backend_tl_concat:\n"
                                "  layer_id %d\n", layer_id));
for(int i = 0; i < input_size; i++) {
  LLVM_DEBUG(
      llvm::errs() << llvm::format("in %d (%d, %d, %d, %d), la_input:%d,"
                                    " rshift:%d multiplier: %d\n",
                                    i, output_dim[0], input_dim_c[i],
                                    output_dim[2], output_dim[3],
                                    la_input[i], rshift, m_i8[i]));
}

LLVM_DEBUG(
      llvm::errs() << llvm::format("la_output:%d\n", la_output));

  ctx.parallel_disable();

  uint32_t out_csize_local = ALIGN(output_dim[2] * output_dim[3], EU_NUM);
  uint32_t n = output_dim[0];
  uint32_t oc = output_dim[1];
  uint32_t h = output_dim[2];
  uint32_t w = output_dim[3];

  for ( int i = 0; i < input_size; i++) {
    cvk_tl_shape_t input_shape = ctx.tl_shape_t4(n,input_dim_c[i],h,w);
    cvk_tl_t tl_input;
    tl_input.start_address = la_input[i];
    tl_input.fmt = CVK_FMT_I8;
    tl_input.shape = input_shape;
    tl_input.stride =
      ctx.tl_default_stride(input_shape, CVK_FMT_I8, 1);
    ctx.apply_qi8(&tl_input, layer_id, do_relu ? 1 : 0, rshift, m_i8[i]);
  }

  int concat_c = 0;
  for (int i = 0; i < input_size; i++) {
    uint32_t out_offset = (concat_c / NPU_NUM) * out_csize_local;
    uint32_t out_addr = (concat_c % NPU_NUM) * LOCAL_MEM_SIZE +
                        la_output
                        + out_offset;

    cvk_tl_shape_t shape = ctx.tl_shape_t4(n,input_dim_c[i],h,w);
    cvk_tl_shape_t out_shape = ctx.tl_shape_t4(n,oc,h,w);

    cvk_tl_t tl_input;
    tl_input.start_address = la_input[i];
    tl_input.fmt = CVK_FMT_I8;
    tl_input.shape = shape;
    tl_input.stride =
      ctx.tl_default_stride(shape, CVK_FMT_I8, 1);

    cvk_tl_t tl_output;
    tl_output.start_address = out_addr;
    tl_output.fmt = CVK_FMT_I8;
    tl_output.shape = shape;
    tl_output.stride =
      ctx.tl_default_stride(out_shape, CVK_FMT_I8, 1);

    cvk_tdma_l2l_tensor_copy_param_t p10 = {0};
    p10.dst = &tl_output;
    p10.src = &tl_input;
    ctx.tdma_l2l_tensor_copy(&p10);

    concat_c += input_dim_c[i];
  }
}

void cvi_backend_tl_bf16_concat(const CviBackendContext &ctx, uint32_t layer_id,
                           int *input_dim_c, int input_size, int *output_dim,
                           laddr_t *la_input, laddr_t la_output, laddr_t la_working) {

  LLVM_DEBUG(
        llvm::errs() << llvm::format("cvi_backend_tl_bf16_concat:\n"
                                  "  layer_id %d\n", layer_id));
  for(int i = 0; i < input_size; i++) {
    LLVM_DEBUG(
        llvm::errs() << llvm::format("in %d (%d, %d, %d, %d), la_input:%d,\n",
                                      i, output_dim[0], input_dim_c[i],
                                      output_dim[2], output_dim[3],
                                      la_input[i]));
  }

  LLVM_DEBUG(
        llvm::errs() << llvm::format("la_output:%d\n", la_output));

  ctx.parallel_disable();

  uint32_t out_csize_local = ALIGN(output_dim[2] * output_dim[3] * sizeof(uint16_t), EU_NUM);
  uint32_t n = output_dim[0];
  uint32_t oc = output_dim[1];
  uint32_t h = output_dim[2];
  uint32_t w = output_dim[3];

  uint32_t concat_c = 0;
  for (int i = 0; i < input_size; i++) {
    uint32_t out_offset = (concat_c / NPU_NUM) * out_csize_local;
    uint32_t out_addr = (concat_c % NPU_NUM) * LOCAL_MEM_SIZE +
                        la_output
                        + out_offset;

    cvk_tl_shape_t shape = ctx.tl_shape_t4(n,input_dim_c[i],h,w);
    cvk_tl_shape_t out_shape = ctx.tl_shape_t4(n,oc,h,w);

    cvk_tl_t tl_input;
    tl_input.start_address = la_input[i];
    tl_input.fmt = CVK_FMT_BF16;
    tl_input.shape = shape;
    tl_input.stride =
      ctx.tl_default_stride(shape, CVK_FMT_BF16, 1);

    cvk_tl_t tl_output;
    tl_output.start_address = out_addr;
    tl_output.fmt = CVK_FMT_BF16;
    tl_output.shape = shape;
    tl_output.stride =
      ctx.tl_default_stride(out_shape, CVK_FMT_BF16, 1);

    cvk_tdma_l2l_tensor_copy_param_t p10 = {0};
    p10.dst = &tl_output;
    p10.src = &tl_input;
    ctx.tdma_l2l_tensor_copy(&p10);

    concat_c += input_dim_c[i];
  }
}
