/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgSliceKernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <cmath>

void cvi_backend_tg_slice_kernel(const CviBackendContext &ctx,
                                 uint32_t layer_id, gaddr_t input_gaddr,
                                 gaddr_t output_gaddr, int input_dim_size,
                                 int *input_dim, int axis, int offset,
                                 int length, cvk_fmt_t fmt) {
  assert(input_dim_size > axis && (offset + length <= input_dim[axis]) &&
         "paramter error");
  assert(input_dim_size <= 4 && "dim size should <= 4");
  int shape[4] = {1, 1, 1, 1};
  for (int i = 0; i < input_dim_size; i++) {
    shape[i] = input_dim[i];
  }
  uint64_t offset_size = ctx.bytesize_of_fmt(fmt);
  for (int i = axis + 1; i < input_dim_size; i++) {
    offset_size *= input_dim[i];
  }
  cvk_tg_stride_t src_stride =
      ctx.tg_default_stride(shape[1], shape[2], shape[3], fmt);
  shape[axis] = length;
  cvk_tg_shape_t dst_shape =
      ctx.tg_shape_t4(shape[0], shape[1], shape[2], shape[3]);
  cvk_tg_stride_t dst_stride = ctx.tg_default_stride(dst_shape, fmt);
  ctx.tdma_g2g_tensor_copy(input_gaddr + offset * offset_size, dst_shape,
                           src_stride, fmt, output_gaddr, dst_shape, dst_stride,
                           fmt);
}
