/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: slice_kernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <cmath>

void cvi_backend_tg_slice_kernel(const CviBackendContext &ctx,
                                uint32_t stream_id, uint32_t inst_id, uint32_t layer_id,
                                const uint32_t *depends, uint32_t depends_len,
                                gaddr_t input_gaddr, gaddr_t output_gaddr,
                                int input_dim_size, int *input_dim, int axis,
                                int offset, int length, cvk_fmt_t fmt) {
  assert(input_dim_size > axis && (offset + length <= input_dim[axis]) &&
         "paramter error");

  int32_t dtype_sz = ((cvk_fmt_t)fmt == CVK_FMT_BF16) ? 2 : 1;
  if (axis == 3) {
    cvk_tg_shape_t shape = {
        (uint32_t)input_dim[0],
        (uint32_t)input_dim[1],
        (uint32_t)input_dim[2],
        (uint32_t)length
    };
    cvk_tg_shape_t orig_shape = {
        (uint32_t)input_dim[0],
        (uint32_t)input_dim[1],
        (uint32_t)input_dim[2],
        (uint32_t)input_dim[3]
    };
    cvk_tg_stride_t d_stride = ctx.tg_default_stride(shape, fmt);
    cvk_tg_stride_t s_stride = ctx.tg_default_stride(orig_shape, fmt);
    ctx.tdma_g2g_tensor_copy(input_gaddr + offset * dtype_sz, shape, s_stride,
                             output_gaddr, shape, d_stride, fmt);
  } else {
    uint32_t former_dim = 1;
    uint32_t later_dim = dtype_sz;
    for (int i = 0; i < axis; i++) {
      former_dim *= input_dim[i];
    }
    for (int i = axis + 1; i < input_dim_size; i++) {
      later_dim *= input_dim[i];
    }
    cvk_tg_shape_t shape = {
        1, (uint32_t)former_dim,
        (uint32_t)length,
        (uint32_t)later_dim
    };
    cvk_tg_stride_t s_stride = {
        (uint32_t)(former_dim * later_dim * input_dim[axis]),
        (uint32_t)(later_dim * input_dim[axis]),
        (uint32_t)(later_dim)
    };
    cvk_tg_stride_t d_stride = ctx.tg_default_stride(shape, CVK_FMT_I8);
    ctx.tdma_g2g_tensor_copy(input_gaddr + offset * later_dim, shape, s_stride,
                             output_gaddr, shape, d_stride, CVK_FMT_I8);
  }
}
