/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TLSlice.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include "backend/backend_tl_api.h"

#define DEBUG_TYPE "tl_slice"

void cvi_backend_tl_slice(
        const CviBackendContext &ctx, uint32_t layer_id,
        int64_t *input_dim, int64_t *output_dim, laddr_t la_input,
        laddr_t la_output, int axis, int offset) {
  ctx.parallel_disable();
  for (int i = 0; i < input_dim[0]; i++) {
    auto n_start_laddr = la_input +
           ctx.lmem_tensor_to_size(1, input_dim[1], input_dim[2], input_dim[3]) * i;

    auto cur_lane_size = ctx.lmem_tensor_to_size(1, 1, input_dim[2], input_dim[3]);
    auto cur_npu_idx = la_input / LOCAL_MEM_SIZE;
    auto cur_laddr = (cur_npu_idx + offset) % NPU_NUM * LOCAL_MEM_SIZE +
              (cur_npu_idx + offset) / NPU_NUM * cur_lane_size + n_start_laddr;

    cvk_tl_shape_t tl_shape = ctx.tl_shape_t4(1,
                                  output_dim[1], output_dim[2], output_dim[3]);
    cvk_tl_t tl_input;
    tl_input.start_address = cur_laddr;
    tl_input.fmt = CVK_FMT_I8;
    tl_input.shape = tl_shape;
    tl_input.stride = ctx.tl_default_stride(tl_shape, CVK_FMT_I8, 1);

    cvk_tl_t tl_output;
    tl_output.start_address = la_output + i * ctx.lmem_tensor_to_size(tl_shape,
                                                 CVK_FMT_I8, true);
    tl_output.fmt = CVK_FMT_I8;
    tl_output.shape = tl_shape;
    tl_output.stride = ctx.tl_default_stride(tl_shape, CVK_FMT_I8, 1);

    cvk_tdma_l2l_tensor_copy_param_t p1 = {0};
    p1.dst = &tl_output;
    p1.src = &tl_input;
    ctx.tdma_l2l_tensor_copy(&p1);
  }
}

void cvi_backend_tl_bf16_slice(
        const CviBackendContext &ctx, uint32_t layer_id,
        int64_t *input_dim, int64_t *output_dim, laddr_t la_input,
        laddr_t la_output, int axis, int offset) {
  ctx.parallel_disable();
  for (int i = 0; i < input_dim[0]; i++) {
    auto n_start_laddr = la_input +
           ctx.lmem_tensor_to_size(1, input_dim[1], input_dim[2], input_dim[3]) * i;

    auto cur_lane_size = ctx.lmem_tensor_to_size(1, 1, input_dim[2],
                                            input_dim[3], CVK_FMT_BF16);
    auto cur_npu_idx = la_input / LOCAL_MEM_SIZE;
    auto cur_laddr = (cur_npu_idx + offset) % NPU_NUM * LOCAL_MEM_SIZE +
              (cur_npu_idx + offset) / NPU_NUM * cur_lane_size + n_start_laddr;

    cvk_tl_shape_t tl_shape = ctx.tl_shape_t4(1,
                                  output_dim[1], output_dim[2], output_dim[3]);
    cvk_tl_t tl_input;
    tl_input.start_address = cur_laddr;
    tl_input.fmt = CVK_FMT_BF16;
    tl_input.shape = tl_shape;
    tl_input.stride = ctx.tl_default_stride(tl_shape, CVK_FMT_BF16, 1);

    cvk_tl_t tl_output;
    tl_output.start_address = la_output + i * ctx.lmem_tensor_to_size(tl_shape,
                                                 CVK_FMT_BF16, true);
    tl_output.fmt = CVK_FMT_BF16;
    tl_output.shape = tl_shape;
    tl_output.stride = ctx.tl_default_stride(tl_shape, CVK_FMT_BF16, 1);

    cvk_tdma_l2l_tensor_copy_param_t p1 = {0};
    p1.dst = &tl_output;
    p1.src = &tl_input;
    ctx.tdma_l2l_tensor_copy(&p1);
  }
}

